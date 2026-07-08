#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ground-truth 실험: YOLO-seg 산출물(물체 점군)을 알려진 회전 R(t)로 만들어
raw / 기존(PoseStabilizer) / 수정(RealtimePCASmoother)의 '진짜 정확도'를 측정.
written by dongmin — 비판자 지적(ground truth 부재 / flip 지표 자기충족 / lag vs accuracy)에 대한 응답 실험.

핵심 질문 3개:
  Q1. 수정본은 raw 대비 '정확도(대칭인지 주축오차)'를 개선하나?  (jitter 말고 GT 대비 오차)
  Q2. 기존 PoseStabilizer의 낮은 jitter는 정확도인가 lag인가?     (ramp+step 응답)
  Q3. 물체가 실제로 180° 돌면 수정본이 그걸 flip으로 '교정'해 버리나?

방법: 가상 depth 카메라 + 부분관측(front-facing 컬링) + 노이즈. YOLO 자체는 미실행(가중치/이미지 없음),
      그 하류(점군→PCA→안정화)는 실제 코드 그대로. 정답 주축 = R(t)·e1.

의존성: numpy + 같은 폴더 pca_multitool_benchmark. 실행: python3 pca_groundtruth_experiment.py
"""
import os, csv, math
import numpy as np
from pca_multitool_benchmark import pca_obb_3d, RealtimePCASmoother, PoseStabilizer

HERE = os.path.dirname(os.path.abspath(__file__))
N = 200
SEED = 424242
NOISE_M = 0.0010
CAM_DEPTH = 0.50   # 물체를 카메라 앞 0.5m


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def box_with_normals(dims, n, rng):
    """6면 박스 표면 점 + 외향 법선."""
    L, a, b = dims
    half = np.array([L/2, a/2, b/2])
    faces = [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1)]
    pts, nrm = [], []
    for ax, sgn in faces:
        k = n // 6
        p = rng.uniform(-1, 1, size=(k, 3)) * half
        p[:, ax] = sgn * half[ax]
        nn = np.zeros((k, 3)); nn[:, ax] = sgn
        pts.append(p); nrm.append(nn)
    return np.vstack(pts).astype(np.float32), np.vstack(nrm).astype(np.float32)


def cyl_with_normals(dims, n, rng):
    """원통 측면 점 + 반경 법선 (주축 = x)."""
    L, a, _ = dims
    r = a/2
    x = rng.uniform(-L/2, L/2, n)
    th = rng.uniform(0, 2*np.pi, n)
    p = np.stack([x, r*np.cos(th), r*np.sin(th)], axis=1)
    nn = np.stack([np.zeros(n), np.cos(th), np.sin(th)], axis=1)
    return p.astype(np.float32), nn.astype(np.float32)


def gt_rotation(t):
    """정답 자세: ramp(등속 yaw) + step(f90 +30°) + flip trap(f130~ 180° about minor)."""
    yaw = 0.5 * t
    if t >= 90:
        yaw += 30.0
    R = Rz(math.radians(yaw))
    if t >= 130:
        R = R @ Ry(math.pi)   # 실제 물리적 180° 뒤집기 (flip trap)
    return R


def make_frame(P0, N0, R, rng):
    """R로 회전 → 카메라 앞 배치 → front-facing만 남김(부분관측) → 노이즈."""
    P = P0 @ R.T
    Nn = N0 @ R.T
    P = P + np.array([0, 0, CAM_DEPTH])
    # 카메라(원점)를 향한 면만: dot(normal, cam - p) > 0  (cam=origin → -p)
    vis = np.sum(Nn * (-P), axis=1) > 0
    Q = P[vis]
    if len(Q) < 50:
        Q = P
    return (Q + rng.normal(0, NOISE_M, size=Q.shape)).astype(np.float32)


def ang(u, v, signed=False):
    d = float(np.dot(u, v))
    if not signed:
        d = abs(d)
    d = max(-1.0, min(1.0, d))
    return math.degrees(math.acos(d))


def run_tool(name, shape, dims, n_pts):
    rng = np.random.default_rng(SEED)
    P0, N0 = (box_with_normals if shape == "box" else cyl_with_normals)(dims, n_pts, rng)
    frames = [(gt_rotation(t), make_frame(P0, N0, gt_rotation(t), rng)) for t in range(N)]

    smoother = RealtimePCASmoother()
    stab = PoseStabilizer(rng=np.random.default_rng(1))
    np.random.seed(SEED & 0x7FFFFFFF)

    rows = []
    e1 = np.array([1.0, 0, 0])
    for t, (R, pts) in enumerate(frames):
        true_major = R @ e1
        _, ax_raw, _ = pca_obb_3d(pts, smoother=None)
        _, ax_s, _ = pca_obb_3d(pts, smoother=smoother)
        _, ax_p0, _ = pca_obb_3d(pts, smoother=None)
        ax_pose = stab.update(ax_p0.copy(), pts)
        rows.append(dict(
            tool=name, frame=t,
            gt_yaw=round(0.5*t + (30 if t >= 90 else 0), 2),
            flip_trap=int(t >= 130),
            n_pts=len(pts),
            # 대칭인지 주축 정확도(선-선 각도, 부호무시) = 진짜 accuracy
            err_raw=round(ang(ax_raw[:, 0], true_major), 4),
            err_pose=round(ang(ax_pose[:, 0], true_major), 4),
            err_smooth=round(ang(ax_s[:, 0], true_major), 4),
            # 부호포함 오차(수정본의 부호강제/기존의 lag가 드러남)
            signed_raw=round(ang(ax_raw[:, 0], true_major, signed=True), 2),
            signed_smooth=round(ang(ax_s[:, 0], true_major, signed=True), 2),
        ))
    return rows


def summarize(rows, name):
    seg = lambda a, b: [r for r in rows if a <= r["frame"] < b]
    mean = lambda rs, k: sum(r[k] for r in rs)/len(rs)
    allr = rows
    steady = seg(20, 88)          # 정상 추종 구간
    settle = seg(90, 105)         # step 직후 (lag 드러남)
    trap = seg(130, 145)          # 180° flip trap 직후
    return dict(
        tool=name,
        acc_raw_mean=round(mean(allr, "err_raw"), 3),
        acc_smooth_mean=round(mean(allr, "err_smooth"), 3),
        acc_pose_mean=round(mean(allr, "err_pose"), 3),
        raw_eq_smooth=all(abs(r["err_raw"]-r["err_smooth"]) < 1e-6 for r in allr),
        steady_raw=round(mean(steady, "err_raw"), 3),
        steady_pose=round(mean(steady, "err_pose"), 3),
        step_pose=round(mean(settle, "err_pose"), 3),   # step 직후 기존 오차↑면 lag
        step_raw=round(mean(settle, "err_raw"), 3),
        trap_signed_raw=round(mean(trap, "signed_raw"), 1),
        trap_signed_smooth=round(mean(trap, "signed_smooth"), 1),
    )


def main():
    tools = [("pliers_box", "box", (0.180, 0.040, 0.020), 2000),
             ("screwdriver_cyl", "cyl", (0.200, 0.020, 0.020), 1800)]
    all_rows, summ = [], []
    for name, shape, dims, n in tools:
        rows = run_tool(name, shape, dims, n)
        all_rows += rows
        summ.append(summarize(rows, name))

    with open(os.path.join(HERE, "gt_per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)

    print("=== Ground-truth 실험: 정답 R(t) 대비 주축 정확도 (jitter 아님) ===\n")
    for s in summ:
        print(f"[{s['tool']}]")
        print(f"  대칭인지 주축오차(평균)  raw={s['acc_raw_mean']}°  수정={s['acc_smooth_mean']}°  기존={s['acc_pose_mean']}°")
        print(f"  → Q1: raw==수정 (전 프레임 동일)? {s['raw_eq_smooth']}  ⇒ 수정본 정확도 개선 = {'0 (부호만 다름)' if s['raw_eq_smooth'] else '있음'}")
        print(f"  → Q2: step(f90) 직후 오차  raw={s['step_raw']}°  기존={s['step_pose']}°  (기존↑면 lag)")
        print(f"  → Q3: flip trap(실제180°) 직후 부호포함오차  raw={s['trap_signed_raw']}°  수정={s['trap_signed_smooth']}°")
        print()

    # ponytail: 핵심 가설 자체검증
    s = summ[0]
    assert s["raw_eq_smooth"], "raw와 수정의 대칭인지 오차가 다름 — 가설(부호정렬=정확도 무관) 재확인 필요"
    print("[self-check OK] 대칭인지 정확도에서 raw==수정 확인 → 수정본은 정확도가 아니라 '부호 연속성'만 제공.")
    print(f"CSV: gt_per_frame.csv ({len(all_rows)}행)")


if __name__ == "__main__":
    main()
