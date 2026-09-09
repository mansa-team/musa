import json
import logging
import os
import random
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset",
    "data",
    "output",
    "corpus-250M.jsonl",
)

BASE_MODEL = "Itau-Unibanco/NorBERTo-base"
DAPT_MODEL = "heitorrosa/logun-base"

N_DOCS = 100
MAX_LEN = 256
MASK_P = 0.15
SEED = 42
BATCH = 8


def sampleTexts(corpusPath: str) -> list[str]:
    rng = random.Random(SEED,)
    picked: list[str] = []
    with open(corpusPath, "r", encoding="utf-8",) as handle:
        for line in handle:
            text = json.loads(line,).get("text", "")
            if not text.strip():
                continue
            if len(picked) < N_DOCS:
                picked.append(text,)
            elif rng.random() < N_DOCS / (len(picked) + 1):
                picked[rng.randrange(N_DOCS,)] = text
    rng.shuffle(picked,)
    return picked[:N_DOCS]


def maskBatch(
    inputIds: torch.Tensor,
    tokenizer: AutoTokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = inputIds.clone()
    rand = torch.rand(inputIds.shape,)
    special = (
        (inputIds == tokenizer.cls_token_id)
        | (inputIds == tokenizer.sep_token_id)
        | (inputIds == tokenizer.pad_token_id)
    )
    masked = (rand < MASK_P) & (~special)
    labels[~masked] = -100
    inputIds[masked] = tokenizer.mask_token_id
    return inputIds, labels


def evalModel(modelPath: str, texts: list[str]) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    snapPath = snapshot_download(modelPath)
    print(f"{modelPath} -> {snapPath}")

    tokenizer = AutoTokenizer.from_pretrained(snapPath)
    model = AutoModelForMaskedLM.from_pretrained(snapPath)
    model.to(device,)
    model.eval()

    torch.manual_seed(SEED)

    totals = {"loss": 0.0, "hits": 0, "masked": 0, "batches": 0}
    with torch.no_grad():
        for start in range(0, len(texts), BATCH):
            chunk = texts[start:start + BATCH]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )

            inputIds = enc["input_ids"]
            inputIds, labels = maskBatch(inputIds, tokenizer)

            out = model(
                inputIds.to(device),
                attention_mask=enc["attention_mask"].to(device),
                labels=labels.to(device),
            )

            totals["loss"] += float(out.loss)

            pred = out.logits.argmax(-1,).cpu()
            keep = labels != -100

            totals["hits"] += int(((pred == labels) & keep).sum())
            totals["masked"] += int(keep.sum())
            totals["batches"] += 1

    loss = totals["loss"] / totals["batches"]
    return {
        "loss": loss,
        "ppl": float(torch.exp(torch.tensor(loss))),
        "acc": totals["hits"] / totals["masked"],
        "nMasked": totals["masked"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    scriptDir = os.path.dirname(os.path.abspath(__file__))
    outPath = os.path.join(scriptDir, "holdout_mlm.json")
    texts = sampleTexts(CORPUS_PATH)

    print(f"Sampled {len(texts)} docs")

    results = {}
    for name, path in (("base", BASE_MODEL), ("dapt", DAPT_MODEL)):
        metrics = evalModel(path, texts)
        results[name] = metrics

        print(f"{name}: loss={metrics['loss']:.4f} ppl={metrics['ppl']:.2f} acc={metrics['acc']:.4f}")

    gap = results["base"]["loss"] - results["dapt"]["loss"]
    print(f"delta loss (base-dapt): {gap:.4f}")

    with open(outPath, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Wrote {outPath}")
