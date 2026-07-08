# -*- coding: utf-8 -*-
"""
실시간 PCA 자세 안정화 (written by dongmin, 2025-08-28)

3D PCA로 구한 물체 주축이 프레임마다 뒤집히거나 바뀌는 문제를 완화하기 위한 모듈이다.
세 가지를 결합한다.
  1) 고유값 비율로 안정/불안정 판정 (check_projection_stability)
  2) 불안정할 때 Power iteration으로 주축만 근사 재계산 (power_iteration)
  3) 직전 프레임과 부호를 맞춰 u 와 -u 뒤집힘 제거 (fast_axis_alignment)

주의: 이 모듈은 프레임 간 부호 연속성을 제공한다. 실험(experiments/)에서 확인했듯이
      대칭 인지 정확도 자체를 올리지는 않는다. 자세한 검증 결과는 저장소 README를 참고.
"""
import numpy as np


class RealtimePCASmoother:
    def __init__(self, stability_thresh=1.5, power_iter_trigger=0.8):
        self.prev_axes = None
        self.frame_count = 0
        self.stability_thresh = stability_thresh   # 고유값 비율 임계치
        self.power_iter_trigger = power_iter_trigger

    def power_iteration(self, cov_matrix, max_iter=3):
        """전체 고유분해 없이 최대 고유벡터(주축)만 반복곱으로 근사한다."""
        n = cov_matrix.shape[0]
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        for _ in range(max_iter):
            v = cov_matrix @ v
            v = v / np.linalg.norm(v)
        eigenval = v.T @ cov_matrix @ v
        return v, eigenval

    def check_projection_stability(self, vals):
        """고유값 비율(l1/l2, l2/l3)이 임계치보다 크면 안정으로 본다."""
        if len(vals) < 3:
            return False
        r1 = vals[0] / vals[1] if vals[1] > 1e-6 else float("inf")
        r2 = vals[1] / vals[2] if vals[2] > 1e-6 else float("inf")
        return r1 > self.stability_thresh and r2 > self.stability_thresh

    def fast_axis_alignment(self, axes_new):
        """새 축과 직전 축의 내적이 음수면 부호를 뒤집어 방향을 맞춘다."""
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
    """
    3D 점군에서 PCA 기반 방향성 경계 박스(OBB)를 계산한다.
    반환: center(3,), axes(3,3) [열벡터 u1,u2,u3 = 장축, 단축, 세번째축], lengths(3,), corners(8,3)
    smoother 를 넘기면 안정화(고유값 게이트 + Power iteration + 부호정렬)를 적용한다.
    """
    pts = points_xyz.astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)

    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axes = vecs[:, order]
    for k in range(3):
        nrm = np.linalg.norm(axes[:, k])
        if nrm > 0:
            axes[:, k] /= nrm

    if smoother is not None:
        smoother.frame_count += 1
        is_stable = smoother.check_projection_stability(vals[order])
        if (not is_stable) and (smoother.frame_count % 5 == 0):
            try:
                main_axis, _ = smoother.power_iteration(cov, max_iter=3)
                axes[:, 0] = main_axis
            except Exception:
                pass
        axes = smoother.fast_axis_alignment(axes)

    proj = centered @ axes
    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)
    c_local = (mins + maxs) * 0.5
    half = (maxs - mins) * 0.5
    center = mean + axes @ c_local

    corners = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            for s3 in (+1, -1):
                corners.append(center + s1*half[0]*axes[:, 0] + s2*half[1]*axes[:, 1] + s3*half[2]*axes[:, 2])
    corners = np.stack(corners, axis=0)
    lengths = 2.0 * half
    return center, axes, lengths, corners
