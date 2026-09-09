import json
import logging
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

SPLIT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset",
    "data",
    "output",
    "split-250M.json",
)
CATEGORIES_PLOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "split_categories.png",
)
TOP_N = 10


def loadSplit(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def topCategories(totals: dict, selected: dict) -> tuple:
    ranked = sorted(totals, key=lambda key: totals[key], reverse=True)

    top = ranked[:TOP_N]
    rest = ranked[TOP_N:]
    top = top + ["other"]

    totalSum = sum(totals.values())
    selSum = sum(selected.values())

    totalShare = [totals[key] / totalSum for key in ranked[:TOP_N]]
    selShare = [selected.get(key, 0) / selSum for key in ranked[:TOP_N]]
    
    totalShare.append(sum(totals[key] for key in rest) / totalSum)
    selShare.append(sum(selected.get(key, 0) for key in rest) / selSum)

    return top, totalShare, selShare


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    split = loadSplit(SPLIT_PATH)
    labels, totalShare, selShare = topCategories(split["by_category_total"], split["by_category_selected"])
    os.makedirs(os.path.dirname(CATEGORIES_PLOT), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    xPos = range(len(labels))
    width = 0.35

    ax.bar([x - width / 2 for x in xPos], totalShare, width, label="full pool")
    ax.bar([x + width / 2 for x in xPos], selShare, width, label="selected")
    ax.set_xticks(list(xPos))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("share")
    ax.set_title("DSIR: full pool vs selected by category")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CATEGORIES_PLOT, dpi=150)
    
    print(f"Saved {CATEGORIES_PLOT}")
