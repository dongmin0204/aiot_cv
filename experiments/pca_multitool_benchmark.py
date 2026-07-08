#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공구 10종 × 3방식 PCA 자세추정 벤치마크 (엑셀 표/차트용 상세 CSV 출력).
written by dongmin — AIoT 로봇팔 CV: raw vs 기존(PoseStabilizer) vs 수정(RealtimePCASmoother).

방식
  - raw   : pca_obb_3d(smoother=None)                — 매 프레임 full EVD, 정렬/안정화 없음 (기준선)
  - 기존  : raw + PoseStabilizer(RANSAC 평면 + SO(3) SLERP)   — tool.py 원본
  - 수정  : pca_obb_3d(smoother=RealtimePCASmoother) — 8/28 문서: 고유값 게이트 + Power iteration + 부호정렬

출력 (같은 폴더)
  tools_meta.csv       : 공구 제원(치수, 형상, 포인트 수, 고유값 λ1:λ2:λ3, 대칭성)
  bench_summary.csv    : 공구×방식 집계 (속도/안정성/속도개선 전부)
  bench_per_frame.csv  : 공구×방식×프레임 원자료 (프레임시간, λ비율, flip, jitter, stable, power_iter)

의존성: numpy(표준). 실행: python3 pca_multitool_benchmark.py
"""
import os
import csv
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLY_DIR = os.path.join(HERE, "bench_tools")
N_FRAMES = 200
REPEATS = 20           # 집계 타이밍용 반복
NOISE_M = 0.0010       # 1mm 센서 노이즈
OUTLIER_P = 0.30       # 프레임당 아웃라이어 삽입 확률
SEED = 20250828

# ------------------------------------------------------------------
# 공구 10종 정의 (실측 규모 m). shape: box(각형) / cyl(원형=단면 퇴화)
# ------------------------------------------------------------------
TOOLS = [
    # id, name, shape, dims(주,부,3축) m, n_points
    ("T01", "long_nose_pliers",  "box", (0.180, 0.040, 0.020), 1800),
    ("T02", "precision_cutter",  "box", (0.150, 0.030, 0.015), 1500),
    ("T03", "vernier_caliper",   "box", (0.200, 0.070, 0.012), 2200),  # 납작
    ("T04", "hex_wrench",        "cyl", (0.100, 0.010, 0.010), 1200),  # 단면 정사각→퇴화
    ("T05", "nipper",            "box", (0.130, 0.050, 0.025), 1600),
    ("T06", "wire_stripper",     "box", (0.160, 0.050, 0.020), 1700),
    ("T07", "screwdriver",       "cyl", (0.200, 0.020, 0.020), 1500),  # 원통→부축 퇴화
    ("T08", "hex_key_small",     "cyl", (0.060, 0.008, 0.008), 900),   # 강한 퇴화
    ("T09", "wire_cutter",       "box", (0.140, 0.045, 0.022), 1600),
    ("T10", "round_file",        "cyl", (0.180, 0.015, 0.015), 1500),  # 원통 줄
]


# ==================================================================
# PLY I/O (ASCII, numpy만)
# ==================================================================
def write_ply(path, pts):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def read_ply(path):
    with open(path) as f:
        lines = f.read().splitlines()
    n = 0
    hdr_end = 0
    for i, ln in enumerate(lines):
        if ln.startswith("element vertex"):
            n = int(ln.split()[-1])
        if ln.strip() == "end_header":
            hdr_end = i + 1
            break
    data = [list(map(float, lines[hdr_end + k].split()[:3])) for k in range(n)]
    return np.asarray(data, dtype=np.float32)


def sample_tool(shape, dims, n, rng):
    L, a, b = dims
    if shape == "box":
        p = np.stack([rng.uniform(-L/2, L/2, n),
                      rng.uniform(-a/2, a/2, n),
                      rng.uniform(-b/2, b/2, n)], axis=1)
    else:  # cyl: 주축 L, 반경 a/2 (원통 → 부축 분산 동일 = 퇴화)
        x = rng.uniform(-L/2, L/2, n)
        th = rng.uniform(0, 2*np.pi, n)
        r = (a/2) * np.sqrt(rng.uniform(0, 1, n))
        p = np.stack([x, r*np.cos(th), r*np.sin(th)], axis=1)
    return p.astype(np.float32)


def ensure_plys():
    os.makedirs(PLY_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for tid, name, shape, dims, n in TOOLS:
        path = os.path.join(PLY_DIR, f"{tid}_{name}.ply")
        if not os.path.exists(path):
            write_ply(path, sample_tool(shape, dims, n, rng))


# ==================================================================
# PCA OBB + RealtimePCASmoother  (8/28 문서 코드 그대로 이식)
# ==================================================================
class RealtimePCASmoother:
    def __init__(self, stability_thresh=1.5, power_iter_trigger=0.8):
        self.prev_axes = None
        self.frame_count = 0
        self.stability_thresh = stability_thresh
        self.power_iter_trigger = power_iter_trigger
        self.used_power_iter = False   # 계측용

    def power_iteration(self, cov_matrix, max_iter=3):
        n = cov_matrix.shape[0]
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        for _ in range(max_iter):
            v = cov_matrix @ v
            v = v / np.linalg.norm(v)
        eigenval = v.T @ cov_matrix @ v
        return v, eigenval

    def check_projection_stability(self, vals):
        if len(vals) < 3:
            return False
        ratio1 = vals[0] / vals[1] if vals[1] > 1e-6 else float("inf")
        ratio2 = vals[1] / vals[2] if vals[2] > 1e-6 else float("inf")
        return ratio1 > self.stability_thresh and ratio2 > self.stability_thresh

    def fast_axis_alignment(self, axes_new):
        if self.prev_axes is None:
            self.prev_axes = axes_new.copy()
            return axes_new
        axes_aligned = axes_new.copy()
        dots = np.sum(axes_new * self.prev_axes, axis=0)
        flip_mask = dots < 0
        axes_aligned[:, flip_mask] *= -1
        self.prev_axes = axes_aligned
        return axes_aligned


def pca_obb_3d(points_xyz, smoother=None):
    """8/28 문서 pca_obb_3d(points, smoother). vals(내림차순)도 함께 반환(계측용)."""
    pts = points_xyz.astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    axes = vecs[:, order]
    for k in range(3):
        nrm = np.linalg.norm(axes[:, k])
        if nrm > 0:
            axes[:, k] /= nrm
    if smoother is not None:
        smoother.frame_count += 1
        smoother.used_power_iter = False
        is_stable = smoother.check_projection_stability(vals)
        if (not is_stable) and (smoother.frame_count % 5 == 0):
            try:
                main_axis, _ = smoother.power_iteration(cov, max_iter=3)
                axes[:, 0] = main_axis
                smoother.used_power_iter = True
            except Exception:
                pass
        axes = smoother.fast_axis_alignment(axes)
    return mean, axes, vals


# ==================================================================
# 기존: PoseStabilizer (tool.py 원본 — 매 프레임 RANSAC + SLERP)
# ==================================================================
def so3_log(R):
    ct = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    th = np.arccos(ct)
    if th < 1e-6:
        return np.zeros(3)
    wh = (R - R.T) / (2.0 * np.sin(th))
    return np.array([wh[2, 1], wh[0, 2], wh[1, 0]]) * th


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-6:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th)*K + (1 - np.cos(th))*(K @ K)


def slerp_SO3(R0, R1, a):
    return R0 @ so3_exp(a * so3_log(R0.T @ R1))


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
    return best_n if best_n[2] >= 0 else -best_n


class PoseStabilizer:
    def __init__(self, alpha_R=0.25, rng=None):
        self.R_prev = None
        self.alpha_R = alpha_R
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
        n = fit_plane_ransac(pts3d, rng=self.rng)
        if n is not None:
            self.z_up_ref = n if self.z_up_ref is None else 0.8*self.z_up_ref + 0.2*n
            axes3 = self.lock_roll_pitch(axes3, self.z_up_ref)
        if self.R_prev is None:
            self.R_prev = axes3.copy()
            return axes3
        R_s = slerp_SO3(self.R_prev, axes3, self.alpha_R)
        self.R_prev = R_s
        return R_s


# ==================================================================
# 프레임 시퀀스 (고정 자세 물체 + 센서 노이즈 + 아웃라이어)
# ==================================================================
def make_sequence(base_pts, n_frames, rng):
    # 물체를 임의 자세로 한 번 회전(월드 정렬 방지)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 2] *= -1
    canon = base_pts @ Q.T
    frames = []
    for _ in range(n_frames):
        f = canon + rng.normal(0, NOISE_M, size=canon.shape)
        if rng.random() < OUTLIER_P:
            k = rng.integers(1, 6)
            out = rng.uniform(-0.15, 0.15, size=(k, 3))
            f = np.vstack([f, out]).astype(np.float32)
        frames.append(f.astype(np.float32))
    return frames


# ==================================================================
# 계측 유틸
# ==================================================================
def signed_axis_delta_deg(a, b):
    """부호 포함 3축 각도변화 (deg) 벡터. u↔-u 뒤집힘은 180°로 잡힘."""
    dots = np.clip(np.sum(a * b, axis=0), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def run_method(frames, method, rng_seed):
    """method: 'raw'|'pose'|'rt'. 프레임별 (axes, dt_ms, vals, stable, power_iter) 반환."""
    rng = np.random.default_rng(rng_seed)
    np.random.seed(rng_seed & 0x7FFFFFFF)  # power_iteration 재현
    smoother = RealtimePCASmoother() if method == "rt" else None
    stab = PoseStabilizer(rng=rng) if method == "pose" else None
    rec = []
    for f in frames:
        t0 = time.perf_counter()
        mean, axes, vals = pca_obb_3d(f, smoother=smoother)
        if method == "pose":
            axes = stab.update(axes, f)
        dt = (time.perf_counter() - t0) * 1e3
        stable = bool(smoother.check_projection_stability(vals)) if smoother else \
                 bool((vals[0]/max(vals[1], 1e-12) > 1.5) and (vals[1]/max(vals[2], 1e-12) > 1.5))
        pit = bool(smoother.used_power_iter) if smoother else False
        rec.append((axes.copy(), dt, vals.copy(), stable, pit))
    return rec


# ==================================================================
# 메인: 벤치 실행 + CSV 3종 출력
# ==================================================================
def main():
    ensure_plys()
    rng = np.random.default_rng(SEED)

    meta_rows, summ_rows, frame_rows = [], [], []
    methods = [("raw", "raw"), ("pose", "기존_PoseStabilizer"), ("rt", "수정_RealtimePCASmoother")]

    for tid, name, shape, dims, n in TOOLS:
        pts = read_ply(os.path.join(PLY_DIR, f"{tid}_{name}.ply"))
        # 고유값 비율(자세추정 난이도 지표)
        c = np.cov(pts - pts.mean(0), rowvar=False)
        ev = np.sort(np.linalg.eigvalsh(c))[::-1]
        r12, r23 = ev[0]/ev[1], ev[1]/ev[2]
        sym = "degenerate" if r23 < 1.5 else ("weak" if r23 < 3 else "clear")
        meta_rows.append(dict(tool_id=tid, name=name, shape=shape,
                              dim_major_m=dims[0], dim_minor_m=dims[1], dim_third_m=dims[2],
                              n_points=len(pts), lam1=ev[0], lam2=ev[1], lam3=ev[2],
                              ratio_l1_l2=round(r12, 3), ratio_l2_l3=round(r23, 3),
                              minor_symmetry=sym))

        frames = make_sequence(pts, N_FRAMES, rng)

        per_method = {}
        for mkey, mlabel in methods:
            # 집계 타이밍: REPEATS 반복 합산
            t0 = time.perf_counter()
            rec = None
            for rp in range(REPEATS):
                rec = run_method(frames, mkey, rng_seed=SEED + hash(tid) % 1000 + rp)
            total_ms_all = (time.perf_counter() - t0) * 1e3
            seq_ms = total_ms_all / REPEATS

            axes_seq = [r[0] for r in rec]
            dts = np.array([r[1] for r in rec])
            stables = np.array([r[3] for r in rec])
            pits = np.array([r[4] for r in rec])

            # 프레임 간 flip/jitter (해당 방식 출력 기준)
            flips, jit_max, jit_axes = [], [], []
            for i in range(1, len(axes_seq)):
                d = signed_axis_delta_deg(axes_seq[i-1], axes_seq[i])
                jit_axes.append(d)
                jit_max.append(d.max())
                flips.append(bool((d > 90).any()))   # 90°↑ = 부호 뒤집힘/축 교환
            jit_max = np.array(jit_max) if jit_max else np.array([0.0])
            flip_arr = np.array(flips) if flips else np.array([False])

            per_method[mkey] = dict(seq_ms=seq_ms, dts=dts, jit_max=jit_max,
                                    flip_arr=flip_arr, stables=stables, pits=pits)

            summ_rows.append(dict(
                tool_id=tid, name=name, method=mlabel, n_points=len(pts),
                minor_symmetry=sym,
                seq_time_ms=round(seq_ms, 4),
                frame_ms_mean=round(dts.mean(), 5),
                frame_ms_median=round(float(np.median(dts)), 5),
                frame_ms_p95=round(float(np.percentile(dts, 95)), 5),
                frame_ms_std=round(dts.std(), 5),
                fps_est=round(1000.0 / dts.mean(), 1),
                flip_count=int(flip_arr.sum()),
                flip_pct=round(100.0 * flip_arr.mean(), 2),
                jitter_mean_deg=round(float(jit_max.mean()), 4),
                jitter_p95_deg=round(float(np.percentile(jit_max, 95)), 4),
                jitter_max_deg=round(float(jit_max.max()), 4),
                stable_frame_pct=round(100.0 * stables.mean(), 2),
                power_iter_frames=int(pits.sum()),
            ))

            for i in range(len(axes_seq)):
                d = jit_axes[i-1] if i >= 1 else np.zeros(3)
                frame_rows.append(dict(
                    tool_id=tid, name=name, method=mlabel, frame=i,
                    frame_ms=round(float(dts[i]), 5),
                    n_points=len(frames[i]),
                    d_axis0_deg=round(float(d[0]), 4),
                    d_axis1_deg=round(float(d[1]), 4),
                    d_axis2_deg=round(float(d[2]), 4),
                    jitter_max_deg=round(float(d.max()), 4),
                    flip=int(bool((d > 90).any())),
                    stable=int(bool(stables[i])),
                    power_iter=int(bool(pits[i])),
                ))

        # 속도개선/안정성개선 (수정 vs 기존, 수정 vs raw) → summary 행에 붙임
        base = per_method["pose"]["seq_ms"]
        rt = per_method["rt"]["seq_ms"]
        raw = per_method["raw"]["seq_ms"]
        for row in summ_rows:
            if row["tool_id"] == tid:
                row["speedup_vs_기존_pct"] = round((base - rt)/base*100, 2)
                row["speedup_vs_raw_pct"] = round((raw - rt)/raw*100, 2)
                fr_raw = int(per_method["raw"]["flip_arr"].sum())
                fr_rt = int(per_method["rt"]["flip_arr"].sum())
                row["flip_reduction_vs_raw"] = fr_raw - fr_rt

    _write_csv(os.path.join(HERE, "tools_meta.csv"), meta_rows)
    _write_csv(os.path.join(HERE, "bench_summary.csv"), summ_rows)
    _write_csv(os.path.join(HERE, "bench_per_frame.csv"), frame_rows)

    # 콘솔 요약(수정 방식 기준)
    print(f"공구 {len(TOOLS)}종 × 3방식 × {N_FRAMES}프레임 (타이밍 {REPEATS}회 반복 평균)")
    print(f"PLY: {PLY_DIR}/  |  CSV 3종: tools_meta / bench_summary / bench_per_frame")
    print("-" * 92)
    print(f"{'tool':<18}{'sym':<11}{'raw ms':>8}{'기존 ms':>9}{'수정 ms':>9}"
          f"{'수정vs기존':>10}{'raw flip':>9}{'수정 flip':>10}")
    for tid, name, *_ in TOOLS:
        r = {row["method"].split("_")[0]: row for row in summ_rows if row["tool_id"] == tid}
        raw_r = next(x for x in summ_rows if x["tool_id"] == tid and x["method"] == "raw")
        pose_r = next(x for x in summ_rows if x["tool_id"] == tid and x["method"].startswith("기존"))
        rt_r = next(x for x in summ_rows if x["tool_id"] == tid and x["method"].startswith("수정"))
        print(f"{name:<18}{rt_r['minor_symmetry']:<11}"
              f"{raw_r['seq_time_ms']:>8.2f}{pose_r['seq_time_ms']:>9.2f}{rt_r['seq_time_ms']:>9.2f}"
              f"{rt_r['speedup_vs_기존_pct']:>9.1f}%{raw_r['flip_count']:>9}{rt_r['flip_count']:>10}")

    # ponytail: 논리 깨지면 실패하는 자체검증
    rt_rows = [x for x in summ_rows if x["method"].startswith("수정")]
    pose_rows = [x for x in summ_rows if x["method"].startswith("기존")]
    assert all(a["seq_time_ms"] < b["seq_time_ms"] for a, b in zip(rt_rows, pose_rows)), \
        "수정이 기존보다 느린 공구 존재 — RANSAC 제거 효과 점검"
    raw_flip = sum(x["flip_count"] for x in summ_rows if x["method"] == "raw")
    rt_flip = sum(x["flip_count"] for x in rt_rows)
    assert rt_flip <= raw_flip, "수정이 raw보다 flip 많음 — 부호정렬 점검"
    print("-" * 92)
    print(f"[self-check OK] 전 공구에서 수정<기존(속도), flip 합계 raw {raw_flip} → 수정 {rt_flip}")


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    # summary는 나중에 추가된 컬럼도 포함되도록 합집합
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
