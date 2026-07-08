# -*- coding: utf-8 -*-
"""
실험 CSV로 README용 차트 3장을 생성한다. (labels in English to avoid font issues)
run: python docs/figures/make_figures.py   (repo root 기준)
deps: numpy, matplotlib
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "..", "experiments")


def load(name):
    with open(os.path.join(EXP, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


# 1) Outlier trimming vs OBB volume inflation
def fig_outlier():
    rows = load("exp1_outlier_volume.csv")
    x = [float(r["outlier_rate"]) * 100 for r in rows]
    no = [float(r["infl_notrim_x"]) for r in rows]
    tr = [float(r["infl_trim_x"]) for r in rows]
    plt.figure(figsize=(7, 4.2))
    plt.plot(x, no, "o-", color="#d4380d", label="No trimming")
    plt.plot(x, tr, "s-", color="#237804", label="Voxel-density trimming")
    plt.yscale("log")
    plt.xlabel("Outlier rate (%)")
    plt.ylabel("OBB volume inflation (x, log scale)")
    plt.title("Outlier trimming keeps the bounding box tight")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "fig_outlier_volume.png"), dpi=150)
    plt.close()


# 2) Ground-truth principal-axis error per frame (pliers_box)
def fig_gt():
    rows = [r for r in load("gt_per_frame.csv") if r["tool"] == "pliers_box"]
    fr = [int(r["frame"]) for r in rows]
    raw = [float(r["err_raw"]) for r in rows]
    pose = [float(r["err_pose"]) for r in rows]
    sm = [float(r["err_smooth"]) for r in rows]
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(fr, pose, color="#8c8c8c", lw=1.4, label="Prior stabilizer (SLERP)")
    plt.plot(fr, raw, color="#1677ff", lw=1.4, label="Raw PCA")
    plt.plot(fr, sm, color="#237804", lw=1.4, label="RealtimePCASmoother")
    plt.axvline(90, color="#bbb", ls="--", lw=0.8)
    plt.text(91, plt.ylim()[1] * 0.9, "step", fontsize=8, color="#888")
    plt.xlabel("Frame")
    plt.ylabel("Principal-axis error vs ground truth (deg)")
    plt.title("Ground-truth accuracy: low jitter of the prior method was lag")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "fig_gt_accuracy.png"), dpi=150)
    plt.close()


# 3) Raw-PCA flip count vs tool symmetry
def fig_flip():
    meta = {r["name"]: r for r in load("tools_meta.csv")}
    raw = [r for r in load("bench_summary.csv") if r["method"] == "raw"]
    raw.sort(key=lambda r: float(meta[r["name"]]["ratio_l2_l3"]))
    names = [r["name"] for r in raw]
    flips = [int(r["flip_count"]) for r in raw]
    colors = ["#d4380d" if r["minor_symmetry"] == "degenerate" else "#1677ff" for r in raw]
    plt.figure(figsize=(8.5, 4.2))
    plt.bar(range(len(names)), flips, color=colors)
    plt.xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
    plt.ylabel("Axis flips over 200 frames (raw PCA)")
    plt.title("Flips are driven by tool geometry (degenerate = red)")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#d4380d"),
               plt.Rectangle((0, 0), 1, 1, color="#1677ff")]
    plt.legend(handles, ["degenerate (cylindrical / square)", "clear (elongated)"])
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "fig_flip_symmetry.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    fig_outlier()
    fig_gt()
    fig_flip()
    print("saved: fig_outlier_volume.png, fig_gt_accuracy.png, fig_flip_symmetry.png")
