#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존(PoseStabilizer: 매 프레임 RANSAC 평면 + SO(3) SLERP)  vs
수정(PCAPoseFilter: 고유값 게이트 + 불안정 시 직전 pose 유지 = 'PCA 반복 안 함')  A/B 벤치마크.
written by dongmin — 목적: 실제 코드(기존 tool.py / 수정 3d_obb.py) 차이의 효과를 재현 가능한 수치로 증명.

- 두 방식 모두 pca_obb_3d 를 매 프레임 호출한다(축 추정).
- 차이는 그 다음 '안정화' 단계:
    * 기존 : PoseStabilizer.update → fit_plane_ransac(100 iters) + slerp_SO3 를 매 프레임 반복
    * 수정 : PCAPoseFilter.update → 고유값 비율로 안정성 판정, 불안정하면 직전 pose 유지(재처리 안 함)
- 측정: 평균 처리시간(속도) + 프레임 간 축 흔들림(jitter, 안정성) + 수정본이 '유지'한 프레임 수.

의존성: numpy 만. 실행: python3 pca_repeat_benchmark.py
"""
import time
import numpy as np


# ===============================================================
# 공통: PCA OBB (수정본 3d_obb.py — eigvals_desc 반환)
# ===============================================================
def pca_obb_3d(points_xyz):
    pts = points_xyz.astype(np.float32)
    mean = pts.mean(axis=0)
    C = np.cov((pts - mean), rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    eigvals_desc = vals[order]
    axes = vecs[:, order]
    axes = axes / (np.linalg.norm(axes, axis=0, keepdims=True) + 1e-9)
    proj = (pts - mean) @ axes
    mins, maxs = proj.min(axis=0), proj.max(axis=0)
    c_local = (mins + maxs) * 0.5
    half = (maxs - mins) * 0.5
    center = mean + axes @ c_local
    lengths = 2.0 * half
    return center, axes, lengths, eigvals_desc


def ensure_right_handed(R):
    R = np.asarray(R, float).copy()
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return R


# ===============================================================
# 기존: PoseStabilizer (tool.py 원본 — RANSAC 평면 + SLERP 매 프레임)
# ===============================================================
def so3_log(R):
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    w_hat = (R - R.T) / (2.0 * np.sin(theta))
    return np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]]) * theta


def so3_exp(w):
    theta = np.linalg.norm(w)
    if theta < 1e-6:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def slerp_SO3(R0, R1, alpha):
    return R0 @ so3_exp(alpha * so3_log(R0.T @ R1))


def fit_plane_ransac(pts, iters=100, tau=0.01, rng=None):
    N = pts.shape[0]
    if N < 50:
        return None
    rng = rng or np.random.default_rng(0)
    best_inl, best_n = 0, None
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        a, b, c = pts[idx]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -np.dot(n, a)
        inl = np.count_nonzero(np.abs(pts @ n + d) < tau)
        if inl > best_inl:
            best_inl, best_n = inl, n
    if best_n is None:
        return None
    return (best_n if best_n[2] >= 0 else -best_n)


class PoseStabilizer:
    """기존 방식: 매 프레임 RANSAC 평면으로 roll/pitch lock + SO(3) SLERP 시간평활."""
    def __init__(self, alpha_R=0.25, use_plane_lock=True, rng=None):
        self.R_prev = None
        self.alpha_R = alpha_R
        self.use_plane_lock = use_plane_lock
        self.z_up_ref = None
        self.rng = rng or np.random.default_rng(0)

    def lock_roll_pitch(self, R, z_up):
        z = z_up / (np.linalg.norm(z_up) + 1e-9)
        x_raw = R[:, 0]
        x = x_raw - np.dot(x_raw, z) * z
        if np.linalg.norm(x) < 1e-6:
            return R
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        return np.stack([x, y, z], axis=1)

    def update(self, axes3, pts3d):
        if self.use_plane_lock:
            n = fit_plane_ransac(pts3d, rng=self.rng)   # ← 매 프레임 RANSAC (무거움)
            if n is not None:
                self.z_up_ref = n if self.z_up_ref is None else 0.8 * self.z_up_ref + 0.2 * n
                axes3 = self.lock_roll_pitch(axes3, self.z_up_ref)
        if self.R_prev is None:
            self.R_prev = axes3.copy()
            return axes3
        R_s = slerp_SO3(self.R_prev, axes3, self.alpha_R)
        self.R_prev = R_s
        return R_s


# ===============================================================
# 수정: PCAPoseFilter (3d_obb.py — 고유값 게이트 + 불안정 시 반복 안 함)
# ===============================================================
class PCAPoseFilter:
    def __init__(self, ratio_thresh=1.5, keep_last_when_unstable=True):
        self.ratio_thresh = float(ratio_thresh)
        self.keep_last = bool(keep_last_when_unstable)
        self.prev_axes = None
        self.held = 0

    @staticmethod
    def _ensure_right_handed(R):
        R = np.asarray(R, float).copy()
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1.0
        return R

    def _align_to_prev(self, axes_new):
        if self.prev_axes is None:
            return axes_new
        axes = axes_new.copy()
        flip = np.sum(axes * self.prev_axes, axis=0) < 0
        axes[:, flip] *= -1
        return axes

    def _is_stable(self, eigvals_desc):
        if len(eigvals_desc) < 3:
            return False
        l1, l2, l3 = [max(1e-12, float(v)) for v in eigvals_desc]
        return (l1 / l2 > self.ratio_thresh) and (l2 / l3 > self.ratio_thresh)

    def update(self, axes, eigvals_desc):
        if not self._is_stable(eigvals_desc) and self.keep_last and self.prev_axes is not None:
            self.held += 1
            return self.prev_axes          # ← 반복 안 함: 직전 pose 유지
        axes = self._align_to_prev(axes)
        axes = self._ensure_right_handed(axes)
        self.prev_axes = axes
        return axes


# ===============================================================
# 합성 시퀀스: 주축은 뚜렷하나 부축이 애매(near-symmetric)한 공구 → 부축 flip 유발
# ===============================================================
def make_tool_points(n, dims, rng):
    half = np.array(dims) / 2.0
    return rng.uniform(-1, 1, size=(n, 3)) * half


def gen_sequence(n_frames=300, n_pts=1500, static_ratio=0.7, seed=0):
    rng = np.random.default_rng(seed)
    # dims: 주축 12cm, 부축 3.3/3.1cm → λ2≈λ3 (부축 애매). 실제 니퍼/커터 손잡이처럼
    # 단면이 거의 정사각인 공구에서 raw PCA 부축 방향이 프레임마다 뒤집히는 상황을 재현.
    base = make_tool_points(n_pts, dims=(0.12, 0.033, 0.031), rng=rng)
    frames = []
    pos = np.array([0.30, 0.0, 0.50])
    for i in range(n_frames):
        if i == 0 or rng.random() > static_ratio:
            pos = pos + rng.normal(0, 0.02, size=3)
        noise = rng.normal(0, 0.0010, size=base.shape)  # 1mm 센서 노이즈
        frames.append((base + pos + noise).astype(np.float32))
    return frames


def axis_jitter_deg(axes_seq):
    """프레임 간 3축 각도변화 최대값의 평균(deg). 부호 포함(abs 안 씀) —
    축이 u↔-u 로 뒤집히면 180°로 잡힌다. 로봇 파지 방향엔 부호가 중요하므로 이게 맞는 지표.
    작을수록 안정."""
    js = []
    for a, b in zip(axes_seq[:-1], axes_seq[1:]):
        dots = np.clip(np.sum(a * b, axis=0), -1.0, 1.0)   # 부호 유지
        js.append(np.degrees(np.arccos(dots)).max())
    return float(np.mean(js)) if js else 0.0


def benchmark(n_frames=300, n_pts=1500, repeats=10, seed=0):
    frames = gen_sequence(n_frames, n_pts, seed=seed)

    # --- raw (안정화 없음): 부축 flip 이 얼마나 심한지 기준선 ---
    raw_axes = []
    for f in frames:
        _, axes, _, _ = pca_obb_3d(f)
        raw_axes.append(ensure_right_handed(axes))

    # --- 기존: PoseStabilizer (매 프레임 RANSAC + SLERP) ---
    t0 = time.perf_counter()
    old_axes = []
    for _ in range(repeats):
        rng = np.random.default_rng(1)
        stab = PoseStabilizer(rng=rng)
        old_axes = []
        for f in frames:
            _, axes, _, _ = pca_obb_3d(f)
            old_axes.append(stab.update(ensure_right_handed(axes), f))
    t_old = (time.perf_counter() - t0) / repeats

    # --- 수정: PCAPoseFilter (고유값 게이트 + 반복 안 함) ---
    t0 = time.perf_counter()
    new_axes, held = [], 0
    for _ in range(repeats):
        filt = PCAPoseFilter()
        new_axes = []
        for f in frames:
            _, axes, _, eig = pca_obb_3d(f)
            new_axes.append(filt.update(axes, eig))
        held = filt.held
    t_new = (time.perf_counter() - t0) / repeats

    speedup = (t_old - t_new) / t_old * 100.0
    print(f"프레임 수                : {n_frames}  (포인트/프레임 {n_pts})")
    print(f"부축 애매(near-symmetric): λ2≈λ3 → raw PCA 부축 뒤집힘 유발 시나리오")
    print("-" * 58)
    print(f"프레임 간 축 흔들림(jitter, deg, 작을수록 안정):")
    print(f"  raw(안정화 없음)       : {axis_jitter_deg(raw_axes):6.2f}°")
    print(f"  기존 PoseStabilizer    : {axis_jitter_deg(old_axes):6.2f}°")
    print(f"  수정 PCAPoseFilter     : {axis_jitter_deg(new_axes):6.2f}°")
    print("-" * 58)
    print(f"평균 처리시간/시퀀스:")
    print(f"  기존 PoseStabilizer    : {t_old*1e3:7.2f} ms  (매 프레임 RANSAC 100 iters)")
    print(f"  수정 PCAPoseFilter     : {t_new*1e3:7.2f} ms  (고유값 게이트)")
    print(f"  속도 개선              : {speedup:5.1f}% 감소")
    print(f"  수정본이 '유지'한 프레임: {held}/{n_frames}  (불안정 → PCA 반복 안 함)")
    return speedup, axis_jitter_deg(raw_axes), axis_jitter_deg(new_axes)


if __name__ == "__main__":
    print("=== 기존(RANSAC 매프레임) vs 수정(고유값 게이트, 반복 안 함) ===")
    speedup, raw_j, new_j = benchmark()

    # ponytail: 논리 깨지면 실패하는 단일 자체검증
    assert speedup > 0, "수정본이 더 느림 — RANSAC 교체 효과 점검"
    assert new_j <= raw_j + 1e-6, "필터가 오히려 흔들림 키움 — 게이트/부호정렬 점검"
    print("\n[self-check OK] 수정본이 더 빠르고(RANSAC 제거) 축 흔들림도 raw 이하로 유지.")
