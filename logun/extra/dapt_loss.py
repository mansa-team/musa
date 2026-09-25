import glob
import json
import logging
import os
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def findTrainerStates(searchRoot: str) -> list[str]:
    pattern = os.path.join(
        searchRoot,
        "**",
        "trainer_state.json",
    )
    found = glob.glob(
        pattern,
        recursive=True,
    )
    found.sort()
    return found


def loadLossCurve(statePath: str) -> tuple[list[float], list[float]]:
    with open(
        statePath,
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle,)
    history = payload.get(
        "log_history",
        [],
    )
    pairs = [
        (
            float(entry["epoch"]),
            float(entry["loss"]),
        )
        for entry in history
        if "epoch" in entry and "loss" in entry
    ]
    pairs.sort(key=lambda pair: pair[0],)
    epochs = [pair[0] for pair in pairs]
    losses = [pair[1] for pair in pairs]
    return epochs, losses


def plotStates(states: list[str], outDir: str) -> str:
    for statePath in states:
        epochs, losses = loadLossCurve(statePath,)
        label = os.path.basename(os.path.dirname(statePath,),)
        plt.plot(
            epochs,
            losses,
            label=label,
        )
    plt.xlabel("epoch",)
    plt.ylabel("loss",)
    plt.title("DAPT training loss (logun-base-250M)",)
    plt.legend()
    plt.tight_layout()
    pngPath = os.path.join(
        outDir,
        "dapt_loss.png",
    )
    plt.savefig(pngPath,)
    plt.close()
    return pngPath


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,)
    scriptDir = os.path.dirname(os.path.abspath(__file__),)
    searchRoot = os.path.dirname(os.path.dirname(os.path.dirname(scriptDir,),),)
    outDir = scriptDir
    os.makedirs(
        outDir,
        exist_ok=True,
    )
    states = [
        path
        for path in findTrainerStates(searchRoot,)
        if os.path.basename(os.path.dirname(path,)) != "checkpoint-8500"
    ]
    if not states:
        print(f"No trainer_state.json found under {searchRoot}")
    else:
        pngPath = plotStates(
            states,
            outDir,
        )
        logger.info(f"Plotted {len(states)} curves to {pngPath}")
        print(f"Wrote {pngPath} from {len(states)} files")
