import csv, os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TSV = os.path.normpath(os.path.join(HERE, "..", "inference", "autoresearch-results.tsv"))

cfgs, vals, notes = [], [], []
with open(TSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        cfgs.append(row["config"])
        try:
            vals.append(float(row["toksPerSec"]))
        except ValueError:
            vals.append(None)
        notes.append(row.get("note", ""))

kept = [(n == "sane" or "new-best" in n) for n in notes]
x_all = list(range(len(cfgs)))
measured = [v for v in vals if v is not None]

best, last = [], None
for v in vals:
    if v is not None:
        last = v if last is None else max(last, v)
    best.append(last)

n_new = sum("new-best" in n for n in notes)
# ponytail: fixed floor compresses the bottom instead of the top cluster
YMIN = 140.0
YMAX = max(measured) + 4.0
GREEN, DARK, GREY, RED = "#0a7a3a", "#064d25", "#555555", "#b00000"
fig, ax = plt.subplots(figsize=(12, 6))
ax.step(x_all, best, where="post", color=GREEN, linewidth=2, label="Running best")
clipped = []  # (x, true value, cfg) below the visible floor
for x, v, c, k in zip(x_all, vals, cfgs, kept):
    if v is None:
        clipped.append((x, None, c))
    elif v < YMIN:
        clipped.append((x, v, c))
    elif not k:
        ax.scatter([x], [v], c=GREY, s=25, zorder=3,
                   label="Discarded" if "Discarded" not in ax.get_legend_handles_labels()[1] else None)
    else:
        ax.scatter([x], [v], c=GREEN, s=55, edgecolors="black",
                   linewidths=0.9, zorder=4,
                   label="Kept" if "Kept" not in ax.get_legend_handles_labels()[1] else None)
        ax.annotate(c, (x, v), xytext=(6, 8), textcoords="offset points",
                    fontsize=8, color=DARK, rotation=25)
for x, v, c in clipped:
    ax.scatter([x], [YMIN + 0.4], c=RED if v is None else GREY, marker="v",
               s=70, zorder=4,
               label=("Load-fail" if v is None else "Discarded (off-scale, value labeled)")
               if len([l for l in ax.get_legend_handles_labels()[1]
                       if "off-scale" in l or "Load-fail" in l]) < 2 else None)
    tag = "load-fail" if v is None else f"{v:.2f}"
    ax.annotate(f"{c}\n{tag}", (x, YMIN + 0.4), xytext=(6, -22),
                textcoords="offset points", fontsize=8,
                color=RED if v is None else GREY, rotation=25)
ax.set_ylim(YMIN, YMAX)
ax.set_xlabel("Experiment #")
ax.set_ylabel("Decode tok/s (higher is better)")
ax.set_title(f"Autoresearch Progress: {len(cfgs)} Experiments, {n_new} Kept Improvements")
ax.text(0.01, 0.02, "Bottom compressed: markers at axis floor show true values below 140 tok/s",
        transform=ax.transAxes, fontsize=7, color=GREY)
ax.grid(alpha=0.35)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "autoresearch-progress.png"), dpi=150)
print("saved")
