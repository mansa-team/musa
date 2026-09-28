"""Regenerate the three SFT paper figures from Hub trainer_state.json data.

Data source: Hub repo heitorrosa/logun-base, run folders sft-0.5/ ... sft-0.8/
(read-only Hub access; latest checkpoint's trainer_state.json per run gives
train history + eval history).

Outputs (into this script's directory, i.e. logun/extra/):
  (1) f1_vs_steps.png — eval macro-F1 vs optimizer steps, LINEAR x
  (2) loss_vs_epochs_smooth.png — rolling-mean-20 train loss/16 + raw + eval
  (3) scaling_best_f1_vs_N.png — best eval-F1 vs train rows, log-scale x

Matplotlib (+numpy) only. Run from anywhere: python logun/extra/plot_sft.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "heitorrosa/logun-base"
RUNS = ["sft-0.5", "sft-0.6", "sft-0.7", "sft-0.8"]
LABELS = {"sft-0.5": "q>=0.5", "sft-0.6": "q>=0.6", "sft-0.7": "q>=0.7", "sft-0.8": "q>=0.8"}
COLORS = {"sft-0.5": "C0", "sft-0.6": "C1", "sft-0.7": "C2", "sft-0.8": "C3"}
# Hardcoded train-row counts per threshold (fixed dataset snapshot).
NROWS = {"sft-0.5": 56999, "sft-0.6": 46853, "sft-0.7": 34476, "sft-0.8": 18682}
GA, W = 16, 20  # grad-accum divisor, rolling-mean window
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


def roll(xs, w):
    return [sum(xs[max(0, i - w + 1):i + 1]) / len(xs[max(0, i - w + 1):i + 1])
            for i in range(len(xs))]


def star(ax, x, y, c, size=15):
    ax.plot(x, y, marker="*", markersize=size, color=c,
            markeredgecolor="black", markeredgewidth=0.7,
            linestyle="None", zorder=5)


def main():
    allf = list_repo_files(REPO)
    data = {}
    for run in RUNS:  # latest checkpoint per run
        nums = sorted({int(p.split("checkpoint-")[1].split("/")[0])
                       for p in allf
                       if p.startswith(run + "/checkpoint-") and "/trainer_state.json" in p})
        for ck in sorted(nums, reverse=True):
            try:
                p = hf_hub_download(REPO, f"{run}/checkpoint-{ck}/trainer_state.json")
                data[run] = json.load(open(p)).get("log_history", [])
                print(f"{run}: ckpt-{ck} n_log={len(data[run])}", flush=True)
                break
            except Exception as e:
                print(f"skip {run}/{ck}: {type(e).__name__}")

    parsed = {}
    for run, hist in data.items():
        tr = sorted([(e["step"], e["loss"] / GA, e.get("epoch")) for e in hist
                     if "loss" in e and "eval_loss" not in e])
        evf = sorted([(e["step"], e.get("eval_macro_f1")) for e in hist
                      if e.get("eval_macro_f1") is not None])
        evl = sorted([(e.get("epoch"), e.get("eval_loss"), e.get("step")) for e in hist
                      if e.get("eval_loss") is not None and e.get("epoch") is not None])
        f1e = sorted([(e.get("epoch"), e.get("eval_macro_f1"), e.get("step")) for e in hist
                      if e.get("eval_macro_f1") is not None])
        best = max(evf, key=lambda t: t[1])
        be = next((t for t in f1e if t[2] == best[0]), None)
        parsed[run] = (tr, evf, evl, best, be)
        print(f"BEST {run}: step={best[0]} macroF1={best[1]:.4f} "
              f"epoch={be[0] if be else None} n_train={len(tr)} n_eval={len(evf)}", flush=True)

    # (1) eval macro-F1 vs optimizer steps, LINEAR x
    fig, ax = plt.subplots()
    for run in RUNS:
        _, evf, _, best, _ = parsed[run]
        xs, ys = zip(*evf)
        ax.plot(xs, ys, marker="o", markersize=3, label=LABELS[run], color=COLORS[run])
        star(ax, best[0], best[1], COLORS[run])
    ax.set_xlabel("Optimizer steps")
    ax.set_ylabel("Eval macro-F1")
    ax.set_title("Eval macro-F1 vs steps (sft q>=0.5/0.6/0.7/0.8)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "f1_vs_steps.png", dpi=150)
    plt.close(fig)
    print("saved f1_vs_steps.png", flush=True)

    # (2) rolling-mean-20 train loss/16 + faint raw + eval_loss markers + best stars, LINEAR y
    fig, ax = plt.subplots()
    for run in RUNS:
        tr, _, evl, _, be = parsed[run]
        c = COLORS[run]
        xe = [t[2] for t in tr]
        yl = [t[1] for t in tr]
        ax.plot(xe, yl, marker="o", markersize=2, linestyle="None", alpha=0.15, color=c)
        ax.plot(xe, roll(yl, W), linewidth=2, label=f"{LABELS[run]} train (mean{W})", color=c)
        ax.plot([t[0] for t in evl], [t[1] for t in evl], marker="s", markersize=4,
                linewidth=1, linestyle="--", label=f"{LABELS[run]} eval", color=c)
        if be is not None:
            m = [t for t in evl if t[2] == be[2]]
            sy = m[0][1] if m else min(evl, key=lambda t: abs(t[0] - be[0]))[1]
            star(ax, be[0], sy, c)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss (train loss_true = logged/16; eval eval_loss)")
    ax.set_title("Train rolling-mean(20) + raw + eval_loss vs epochs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "loss_vs_epochs_smooth.png", dpi=150)
    plt.close(fig)
    print("saved loss_vs_epochs_smooth.png", flush=True)

    # (3) best eval-F1 vs train rows, log-scale x, log-linear fit on first three
    xs3 = [math.log10(NROWS[r]) for r in RUNS[:3]]
    ys3 = [parsed[r][3][1] for r in RUNS[:3]]
    n = len(xs3)
    sx, sy = sum(xs3), sum(ys3)
    sxx = sum(x * x for x in xs3)
    sxy = sum(x * y for x, y in zip(xs3, ys3))
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    print(f"trend: F1 = {a:.4f} + {b:.4f}*log10(N)", flush=True)
    champ = max(RUNS, key=lambda r: parsed[r][3][1])
    fig, ax = plt.subplots()
    xg = np.logspace(math.log10(min(NROWS.values())) * 0.995,
                     math.log10(max(NROWS.values())) * 1.002, 200)
    ax.plot(xg, [a + b * math.log10(v) for v in xg], linestyle="--",
            linewidth=1.5, alpha=0.6, color="gray", label="log-linear fit (0.5-0.7)")
    lo = min(NROWS[r] for r in RUNS[:3])
    xe2 = [v for v in xg if v < lo]
    ax.plot(xe2, [a + b * math.log10(v) for v in xe2], linestyle=":",
            linewidth=1.5, alpha=0.35, color="gray")
    for run in RUNS:
        x, y = NROWS[run], parsed[run][3][1]
        ax.plot(x, y, marker="o", markersize=7, color=COLORS[run],
                label=f"{LABELS[run]} (N={x}, F1={y:.4f})")
        ax.annotate(LABELS[run], (x, y), textcoords="offset points", xytext=(6, 8), fontsize=8)
    ax.plot(NROWS[champ], parsed[champ][3][1], marker="*", markersize=18,
            color="gold", markeredgecolor="black", markeredgewidth=0.8,
            linestyle="None", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Train rows N (log-scale x)")
    ax.set_ylabel("Best eval macro-F1")
    ax.set_title(f"Scaling: best eval macro-F1 vs train rows (champion: {LABELS[champ]})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "scaling_best_f1_vs_N.png", dpi=150)
    plt.close(fig)
    print("saved scaling_best_f1_vs_N.png", flush=True)


if __name__ == "__main__":
    main()
