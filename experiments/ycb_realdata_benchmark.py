# -*- coding: utf-8 -*-
"""
실제 물체 지오메트리(YCB 공구 메시) + 통제된 회전 모션으로 자세 안정화 방식 비교.
written by dongmin.

합성 도형(box/원통) 대신 실제 공구 메시(Hammer, PowerDrill, Scissors, MediumClamp)를 표면 샘플링해
알려진 회전 R(t)로 움직이며(부분관측 + 노이즈) raw / 기존(PoseStabilizer) / 수정(RealtimePCASmoother)을 비교한다.

핵심 지표:
  1) 프레임 간 축 뒤집힘(flip, 90도 초과) 횟수  <- 정답 pose 없이도 재는 시간 일관성. 안정화의 실제 역할.
  2) 프레임 간 jitter(부호 포함 각변화)
  (참고) 정답 주축 대비 대칭인지 오차 - 단 PCA-OBB 축은 CAD 자세 프레임이 아니므로 근사 지표.

주의: 자세 안정화는 시간축 문제라 물체가 움직이는 시퀀스에서만 의미가 있다.
의존성: numpy(표준) + 같은 폴더 pca_multitool_benchmark. 메시는 최초 실행 시 자동 다운로드(urllib).
실행: python3 ycb_realdata_benchmark.py
"""
import os, csv, math, urllib.request
import numpy as np
from pca_multitool_benchmark import pca_obb_3d, RealtimePCASmoother, PoseStabilizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "ycb_data")
N = 200
NOISE_M = 0.0015
CAM_DEPTH = 0.6
SEED = 7

BASE = "https://raw.githubusercontent.com/eleramp/pybullet-object-models/master/pybullet_object_models/ycb_objects"
TOOLS = {
    "hammer": f"{BASE}/YcbHammer/textured_reoriented.obj",
    "power_drill": f"{BASE}/YcbPowerDrill/textured_simple_reoriented.obj",
    "scissors": f"{BASE}/YcbScissors/textured_simple_reoriented.obj",
    "medium_clamp": f"{BASE}/YcbMediumClamp/textured_reoriented.obj",
}


def ensure_mesh(name, url):
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, name + ".obj")
    if not os.path.exists(p) or os.path.getsize(p) < 1000:
        urllib.request.urlretrieve(url, p)
    return p


def load_obj(p):
    V, F = [], []
    for ln in open(p, encoding="utf-8", errors="ignore"):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            idx = [int(t.split("/")[0]) for t in ln.split()[1:]]
            for k in range(1, len(idx) - 1):
                F.append([idx[0] - 1, idx[k] - 1, idx[k + 1] - 1])
    return np.array(V, float), np.array(F, int)


def sample_surface(V, F, n, rng):
    """면적 가중 표면 샘플링 + 면 법선. 메시 중심을 원점으로."""
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    nrm = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)
    prob = area / area.sum()
    fi = rng.choice(len(F), size=n, p=prob)
    r1 = np.sqrt(rng.uniform(0, 1, n)); r2 = rng.uniform(0, 1, n)
    a = (1 - r1)[:, None]; b = (r1 * (1 - r2))[:, None]; c = (r1 * r2)[:, None]
    pts = a * v0[fi] + b * v1[fi] + c * v2[fi]
    pts = pts - V.mean(0)  # center
    return pts.astype(np.float64), nrm[fi].astype(np.float64)


def Rz(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def gt_rotation(t):
    yaw = 0.6 * t + (30 if t >= 90 else 0)   # ramp + step
    return Rz(math.radians(yaw))


def make_frame(P0, N0, R, rng):
    P = P0 @ R.T
    Nn = N0 @ R.T
    P = P + np.array([0, 0, CAM_DEPTH])
    vis = np.sum(Nn * (-P), axis=1) > 0        # front-facing only (partial view)
    Q = P[vis]
    if len(Q) < 50:
        Q = P
    return (Q + rng.normal(0, NOISE_M, Q.shape)).astype(np.float64)


def signed_axis_delta(a, b):
    return math.degrees(math.acos(max(-1, min(1, float(a @ b)))))


def sym_err(a, gt):
    return math.degrees(math.acos(min(1, abs(float(a @ gt)))))


def run_tool(name, url, rng):
    P0, N0 = sample_surface(*load_obj(ensure_mesh(name, url)), n=3000, rng=rng)
    e1 = np.array([1.0, 0, 0])
    # 정답 주축 = 전체 점군 공분산 최대 고유벡터 (canonical)
    C = np.cov(P0 - P0.mean(0), rowvar=False)
    w, v = np.linalg.eigh(C)
    major_canon = v[:, np.argmax(w)]

    frames = [(gt_rotation(t), make_frame(P0, N0, gt_rotation(t), rng)) for t in range(N)]
    smoother = RealtimePCASmoother()
    stab = PoseStabilizer(rng=np.random.default_rng(1))
    np.random.seed(SEED)

    rows = []
    prev = {"raw": None, "pose": None, "rt": None}
    for t, (R, pts) in enumerate(frames):
        gt_major = R @ major_canon
        _, ax_raw, _ = pca_obb_3d(pts, smoother=None)
        _, ax_s, _ = pca_obb_3d(pts, smoother=smoother)
        _, ax_p0, _ = pca_obb_3d(pts, smoother=None)
        ax_pose = stab.update(ax_p0.copy(), pts)
        cur = {"raw": ax_raw[:, 0], "pose": ax_pose[:, 0], "rt": ax_s[:, 0]}
        row = dict(tool=name, frame=t, n_pts=len(pts))
        for m in ("raw", "pose", "rt"):
            row[f"{m}_flip"] = int(prev[m] is not None and signed_axis_delta(prev[m], cur[m]) > 90)
            row[f"{m}_jit"] = round(signed_axis_delta(prev[m], cur[m]), 3) if prev[m] is not None else 0.0
            row[f"{m}_symerr"] = round(sym_err(cur[m], gt_major), 3)
            prev[m] = cur[m].copy()
        rows.append(row)
    return rows


def main():
    rng = np.random.default_rng(SEED)
    all_rows = []
    print("=== YCB 실제 공구 메시 + 통제 모션 벤치마크 ===")
    print(f"{'tool':<14}{'raw flip':>9}{'기존 flip':>10}{'수정 flip':>10}{'raw symerr':>12}{'수정 symerr':>12}")
    summ = []
    for name, url in TOOLS.items():
        rows = run_tool(name, url, rng)
        all_rows += rows
        f = lambda m: sum(r[f"{m}_flip"] for r in rows)
        se = lambda m: sum(r[f"{m}_symerr"] for r in rows) / len(rows)
        print(f"{name:<14}{f('raw'):>9}{f('pose'):>10}{f('rt'):>10}{se('raw'):>11.2f}°{se('rt'):>11.2f}°")
        summ.append(dict(tool=name, raw_flip=f("raw"), pose_flip=f("pose"), rt_flip=f("rt"),
                         raw_symerr=round(se("raw"), 3), pose_symerr=round(se("pose"), 3), rt_symerr=round(se("rt"), 3)))

    with open(os.path.join(HERE, "ycb_per_frame.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    with open(os.path.join(HERE, "ycb_summary.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

    # 차트
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        names = [s["tool"] for s in summ]; x = np.arange(len(names)); wd = 0.27
        plt.figure(figsize=(8, 4.2))
        plt.bar(x - wd, [s["raw_flip"] for s in summ], wd, label="raw", color="#1677ff")
        plt.bar(x, [s["pose_flip"] for s in summ], wd, label="prior (PoseStabilizer)", color="#8c8c8c")
        plt.bar(x + wd, [s["rt_flip"] for s in summ], wd, label="ours (RealtimePCASmoother)", color="#237804")
        plt.xticks(x, names, rotation=15); plt.ylabel("axis flips over 200 frames")
        plt.title("Real YCB tool geometry: axis flips per method")
        plt.legend(); plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(HERE, "..", "docs", "figures", "fig_ycb_flip.png"), dpi=150)
        print("chart: docs/figures/fig_ycb_flip.png")
    except Exception as e:
        print("chart skipped:", e)

    # 자체검증
    raw_tot = sum(s["raw_flip"] for s in summ); rt_tot = sum(s["rt_flip"] for s in summ)
    assert rt_tot <= raw_tot, "수정이 raw보다 flip 많음"
    print(f"\n[self-check OK] 실제 공구에서도 flip: raw {raw_tot} -> 수정 {rt_tot}")
    print("CSV: ycb_per_frame.csv, ycb_summary.csv")


if __name__ == "__main__":
    main()
