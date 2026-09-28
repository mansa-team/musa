"""Plot live hero-extension run (Hub heitorrosa/logun-base folder sft/). Read-only, single pass."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "heitorrosa/logun-base"
RUN = "sft"
GA, W = 16, 20
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


def roll(xs, w):
    return [sum(xs[max(0, i - w + 1):i + 1]) / len(xs[max(0, i - w + 1):i + 1])
            for i in range(len(xs))]


def star(ax, x, y, c="red", size=15):
    ax.plot(x, y, marker="*", markersize=size, color=c,
            markeredgecolor="black", markeredgewidth=0.7,
            linestyle="None", zorder=5)


def main():
    allf = list_repo_files(REPO)
    nums = sorted({int(p.split("checkpoint-")[1].split("/")[0])
                   for p in allf
                   if p.startswith(RUN + "/checkpoint-") and "/trainer_state.json" in p})
    assert nums, "no checkpoints found under sft/"
    hist = None
    ck_used = None
    for ck in sorted(nums, reverse=True):  # single read pass, latest first
        try:
            p = hf_hub_download(REPO, f"{RUN}/checkpoint-{ck}/trainer_state.json")
            hist = json.load(open(p)).get("log_history", [])
            ck_used = ck
            print(f"sft: ckpt-{ck} n_log={len(hist)}", flush=True)
            break
        except Exception as e:
            print(f"skip sft/{ck}: {type(e).__name__}")
    assert hist, "failed to read hero trainer_state.json"
    tr = sorted([(e["step"], e["loss"] / GA, e.get("epoch")) for e in hist
                 if "loss" in e and "eval_loss" not in e])
    evf = sorted([(e["step"], e.get("eval_macro_f1")) for e in hist
                  if e.get("eval_macro_f1") is not None])
    evl = sorted([(e.get("epoch"), e.get("eval_loss"), e.get("step")) for e in hist
                  if e.get("eval_loss") is not None and e.get("epoch") is not None])
    best = max(evf, key=lambda t: t[1])
    print(f"BEST hero(ckpt-{ck_used}): step={best[0]} macroF1={best[1]:.4f} "
          f"n_train={len(tr)} n_eval={len(evf)}", flush=True)
    print(f"MIN train_loss_true={min(t[1] for t in tr):.4f} "
          f"min_eval_loss={min(t[1] for t in evl):.4f}" if evl else "", flush=True)

    fig, ax = plt.subplots()  # (a) F1 vs steps, linear x
    xs, ys = zip(*evf)
    ax.plot(xs, ys, marker="o", markersize=3, color="C2")
    star(ax, best[0], best[1], "C2")
    ax.set_xlabel("Optimizer steps")
    ax.set_ylabel("Eval macro-F1")
    ax.set_title(f"Hero sft (q>=0.7 resumed, ckpt-{ck_used}): eval macro-F1 vs steps")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "hero_f1_vs_steps.png", dpi=150)
    plt.close(fig)
    print("saved hero_f1_vs_steps.png", flush=True)

    fig, ax = plt.subplots()  # (b) loss vs epochs, linear y
    xe = [t[2] for t in tr]
    yl = [t[1] for t in tr]
    ax.plot(xe, yl, marker="o", markersize=2, linestyle="None", alpha=0.15, color="C2")
    ax.plot(xe, roll(yl, W), linewidth=2, color="C2", label=f"train (mean{W})")
    ax.plot([t[0] for t in evl], [t[1] for t in evl], marker="s", markersize=4,
            linewidth=1, linestyle="--", color="C2", label="eval")
    m = [t for t in evl if t[2] == best[0]]
    if m:
        star(ax, m[0][0], m[0][1], "C2")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss (train loss_true = logged/16; eval eval_loss)")
    ax.set_title(f"Hero sft: train rolling-mean({W}) + raw + eval_loss vs epochs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "hero_loss_vs_epochs_smooth.png", dpi=150)
    plt.close(fig)
    print("saved hero_loss_vs_epochs_smooth.png", flush=True)


if __name__ == "__main__":
    main()
