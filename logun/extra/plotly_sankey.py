"""Clean layered Sankeys (plotly, marin-style) for the SFT + DAPT pipelines.

Reads DAPT numbers live from logun/dataset/data/output/*.json; SFT counts
are baked-in run records (translate/ data files are not on disk).
Writes logun/extra/sft_flow.png + logun/extra/dapt_flow.png.
"""
import json
import logging
import os

import plotly.graph_objects as go

logger = logging.getLogger(__name__)

EXTRA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(EXTRA_DIR), "dataset", "data", "output")
SFT_PNG = os.path.join(EXTRA_DIR, "sft_flow.png")
DAPT_PNG = os.path.join(EXTRA_DIR, "dapt_flow.png")

BLUE = "rgba(70, 130, 180, 0.55)"
GREEN = "rgba(84, 160, 110, 0.55)"
AMBER = "rgba(220, 160, 60, 0.55)"
GREY = "rgba(150, 150, 150, 0.45)"
RED = "rgba(200, 90, 90, 0.45)"


INK = "#2f6fb2"          # main-stream blue (marin-style monochrome)
PAPER_LOSS = "#9aa0a6"    # loss / side-stream grey
KEEP = "#3d9970"          # deliverable green
DROP = "#c46a5a"          # rejected red


def sankey(nodes, xs, ys, sources, targets, values, colors, nodeColors,
           title, out):
    fig = go.Figure(go.Sankey(
        arrangement="freeform",
        node=dict(label=nodes, x=xs, y=ys, pad=22, thickness=20,
                  color=nodeColors,
                  line=dict(color="rgba(0,0,0,0)", width=0)),
        link=dict(source=sources, target=targets, value=values, color=colors),
    ))
    fig.update_layout(title=title, font=dict(size=13),
                      margin=dict(l=10, r=10, t=50, b=10),
                      width=1200, height=700)
    fig.write_image(out, scale=2)
    print(f"Saved {out}")


def sftFigure():
    # Run-record counts (translate/ data files not on disk; values conserved).
    sft, fgs, twt, fgh = 55484, 38091, 10038, 522
    pool = sft + fgs + twt + fgh          # 104135
    translated = 64692
    dupes = pool - translated             # 39443 covered by canon match
    final, echoes = 63162, 1523
    unmatched = translated - final - echoes   # 7 rows, no score match
    kept, below = 56300, final - 56300    # quality>=0.5 estimate
    nodes = ["SFT 55,484", "Fingpt-sentiment 38,091", "twitter 10,038",
             "FinGPT-headline 522", f"clean pool {pool:,}",
             f"translated {translated:,}", f"dupes (canon match) {dupes:,}",
             f"scored {translated:,}",
             f"final {final:,}", f"echoes+unmatched {echoes + unmatched:,}",
             f"kept (q>=0.5) ~{kept:,}", f"below ~{below:,}"]
    xs = [0.0, 0.0, 0.0, 0.0, 0.22, 0.44, 0.62, 0.62, 0.80, 0.80, 1.0, 1.0]
    ys = [0.16, 0.52, 0.76, 0.90, 0.32, 0.32, 0.88, 0.32, 0.28, 0.62, 0.20, 0.52]
    sources = [0, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7]
    targets = [4, 4, 4, 4, 5, 6, 7, 8, 9, 10, 11]
    values = [sft, fgs, twt, fgh, translated, dupes, translated,
              final, echoes + unmatched, kept, below]
    linkColors = [BLUE] * 5 + [GREY, BLUE, BLUE, GREY, GREEN, RED]
    nodeColors = [INK] * 8 + [PAPER_LOSS, KEEP, DROP]
    sankey(nodes, xs, ys, sources, targets, values, linkColors, nodeColors,
           "SFT pipeline (rows) — concat, dedup, translate, score, 0.5 cut",
           SFT_PNG)


def daptFigure():
    with open(os.path.join(OUT_DIR, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(os.path.join(OUT_DIR, "split-250M.json"), encoding="utf-8") as handle:
        split = json.load(handle)
    docs = manifest.get("documents_loaded", 189788)
    chunks = manifest["chunks_generated"]
    retained = manifest["chunks_retained"]
    filtered = chunks - retained
    sel = split["chunks_selected"]
    notsel = retained - sel
    cats = split["by_category_selected"]
    ranked = sorted(cats, key=cats.get, reverse=True)
    top, rest = ranked[:5], ranked[5:]
    other = sum(cats[k] for k in rest)

    nodes = ([f"CVM docs {docs:,}", f"chunked {chunks:,}",
              f"filtered {filtered:,}", f"retained {retained:,}",
              f"not selected {notsel:,}", f"corpus-250M {sel:,}"]
             + [f"{k} {cats[k]:,}" for k in top] + [f"other {other:,}"])
    n = len(nodes)
    xs = [0.0, 0.22, 0.22, 0.44, 0.44, 0.66] + [1.0] * (n - 6)
    ys = [0.30, 0.34, 0.92, 0.34, 0.92, 0.30] + [0.08, 0.24, 0.40, 0.56, 0.72, 0.88]
    sources = [0, 1, 1, 3, 3, 5, 5, 5, 5, 5, 5]
    targets = [1, 2, 3, 4, 5] + list(range(6, n))
    values = ([chunks, filtered, retained, notsel, sel]
              + [cats[k] for k in top] + [other])
    linkColors = [BLUE, GREY, BLUE, GREY, BLUE] + [BLUE] * 5 + [AMBER]
    nodeColors = [INK, INK, PAPER_LOSS, INK, PAPER_LOSS, INK] + [INK] * 5 + [PAPER_LOSS]
    sankey(nodes, xs, ys, sources, targets, values, linkColors, nodeColors,
           "DAPT pipeline (chunks) — chunk, filter, DSIR-select, by category",
           DAPT_PNG)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sftFigure()
    daptFigure()
