import os
import csv
import json
import logging
import torch
from flashlib.applications.logistic_regression import LogisticRegression as FlashLogisticRegression
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HUB_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

load_dotenv()
logger = logging.getLogger(__name__)

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset",
    "financial_phrase_bank_pt_br.csv",
)

TOKEN = os.environ.get("HF_TOKEN")

SEEDS = (42, 43, 44, 45, 46)
FOLDS = 5
N_CLASSES = 3
PROBE_C = 1.0
MAX_LEN = 256
BATCH = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_COLS = ("text_pt", "text", "sentence", "frase", "sentenca", "sentença")
LABEL_COLS = ("y", "label", "sentiment", "sentimento", "rotulo", "rótulo")
LABEL_MAP = {
    "negativo": 0,
    "negative": 0,
    "neutro": 1,
    "neutral": 1,
    "positivo": 2,
    "positive": 2,
}

MODEL_POOL = (
    "Itau-Unibanco/NorBERTo-base",
    "heitorrosa/logun-base",
    "higopires/DeB3RTa-base",
    "lucas-leme/FinBERT-PT-BR",
)

def loadDataset(path: str) -> tuple:
    with open(path, "r", encoding="latin-1") as handle:
        rows = list(csv.DictReader(handle))

    cols = {key.lower(): key for key in rows[0].keys()}
    textCol = next(cols[key] for key in TEXT_COLS if key in cols)
    labelCol = next(cols[key] for key in LABEL_COLS if key in cols)

    texts = []
    labels = []
    for row in rows:
        label = LABEL_MAP.get(row[labelCol].strip().lower())
        if row[textCol].strip() and label is not None:
            texts.append(row[textCol])
            labels.append(label)

    return texts, labels


def encodeTexts(modelName: str, texts: list) -> list:
    tokenizer = AutoTokenizer.from_pretrained(modelName, token=TOKEN)
    model = AutoModel.from_pretrained(modelName, token=TOKEN)

    model.eval()
    model.to(DEVICE)

    print(f"encode {modelName} on {DEVICE}")

    embs = []
    total = (len(texts) + BATCH - 1) // BATCH
    step = max(total // 10, 1)
    with torch.no_grad():
        for batchIdx, start in enumerate(range(0, len(texts), BATCH)):
            batch = texts[start:start + BATCH]
            if batchIdx % step == 0:
                print(f"encode {modelName}: batch {batchIdx + 1}/{total}")

            tok = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            tok = {key: val.to(DEVICE) for key, val in tok.items()}

            hidden = model(**tok).last_hidden_state
            mask = tok["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            
            embs.extend(pooled.cpu().tolist())

    return embs


def macroF1(pred: torch.Tensor, gold: torch.Tensor) -> float:
    scores = []
    for cls in range(N_CLASSES):
        tp = ((pred == cls) & (gold == cls)).sum().item()
        fp = ((pred == cls) & (gold != cls)).sum().item()
        fn = ((pred != cls) & (gold == cls)).sum().item()
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return sum(scores) / len(scores)


def stratifiedFolds(gold: torch.Tensor, seed: int) -> list:
    gen = torch.Generator().manual_seed(seed)

    folds = [[] for _ in range(FOLDS)]
    for cls in range(N_CLASSES):
        idx = (gold == cls).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx), generator=gen)]
        for pos, row in enumerate(perm.tolist()):
            folds[pos % FOLDS].append(row)

    return folds


def foldScores(feats: torch.Tensor, gold: torch.Tensor, seed: int) -> tuple:
    folds = stratifiedFolds(gold, seed)
    accs = []
    f1s = []
    for fold in range(FOLDS):
        testIdx = torch.tensor(folds[fold], device=DEVICE)
        others = [row for fNum, bucket in enumerate(folds) if fNum != fold for row in bucket]
        trainIdx = torch.tensor(others, device=DEVICE)

        torch.manual_seed(seed + fold)

        mean = feats[trainIdx].mean(0, keepdim=True)
        std = feats[trainIdx].std(0, keepdim=True).clamp_min(1e-6)

        trainFeats = (feats[trainIdx] - mean) / std
        testFeats = (feats[testIdx] - mean) / std
        trainLabels = gold[trainIdx]
        testLabels = gold[testIdx]

        probs = torch.zeros(len(testIdx), N_CLASSES, device=DEVICE)
        for cls in range(N_CLASSES):
            clf = FlashLogisticRegression(C=PROBE_C)
            clf.fit(trainFeats, (trainLabels == cls).long())
            proba = torch.as_tensor(clf.predict_proba(testFeats), device=DEVICE)
            probs[:, cls] = proba[:, 1]

        pred = probs.argmax(1)

        accs.append((pred == testLabels).float().mean().item())
        f1s.append(macroF1(pred, testLabels))
    return sum(accs) / len(accs), sum(f1s) / len(f1s)


if __name__ == "__main__":
    texts, labels = loadDataset(DATASET_PATH)
    print(f"Loaded {len(texts)} labeled sentences from {DATASET_PATH}")

    results = {}
    for name in MODEL_POOL:
        try:
            print(f"probing {name}")
            feats = torch.tensor(encodeTexts(name, texts), device=DEVICE)
            gold = torch.tensor(labels, device=DEVICE)

            accRuns = []
            f1Runs = []
            for seed in SEEDS:
                acc, f1 = foldScores(feats, gold, seed)
                print(f"seed {seed}: acc={acc:.4f} f1={f1:.4f}")
                accRuns.append(acc)
                f1Runs.append(f1)

            results[name] = {
                "accMean": sum(accRuns) / len(accRuns),
                "f1Mean": sum(f1Runs) / len(f1Runs),
                "n": len(texts),
            }

            print(f"{name}: acc={results[name]['accMean']:.4f} f1={results[name]['f1Mean']:.4f} (n={len(texts)})")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"{name}: ERROR {e}")
            continue

    scriptDir = os.path.dirname(os.path.abspath(__file__))
    outPath = os.path.join(scriptDir, "sentiment_probe.json")

    with open(outPath, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved {outPath}")
