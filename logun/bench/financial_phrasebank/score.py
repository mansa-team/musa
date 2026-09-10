import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "models"))
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HUB_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

import csv
import re
import time
import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
from flashlib.applications.logistic_regression import LogisticRegression as FlashLogisticRegression

load_dotenv()

TOKEN = os.environ.get("HF_TOKEN")
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "financial_phrase_bank_pt_br.csv")
SEED = 42
TRAIN_FRAC = 0.8
N_CLASSES = 3
PROBE_C = 1.0
MAX_LEN = 256
BATCH = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_COLS = ("text_pt", "text", "sentence", "frase", "sentenca", "sentença",)
LABEL_COLS = ("y", "label", "sentiment", "sentimento", "rotulo", "rótulo",)
LABEL_MAP = {
    "negativo": 0,
    "negative": 0,
    "neutro": 1,
    "neutral": 1,
    "positivo": 2,
    "positive": 2,
}
ZS_PROMPT = ("Classify the sentiment of this financial sentence as positive, neutral, or negative.\n"
             "Sentence: {text}\n"
             "Answer with one word:")
ZS_LABEL_MAP = {
    "negative": 0,
    "negativo": 0,
    "neutral": 1,
    "neutro": 1,
    "positive": 2,
    "positivo": 2,
}
ZS_PATTERN = re.compile(r"positive|positivo|neutral|neutro|negative|negativo", re.IGNORECASE)


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


def loadEncoder(modelId: str, configOverrides: dict | None = None) -> tuple:
    from transformers import AutoConfig
    print(f"encode {modelId} on {DEVICE}")
    if configOverrides:
        cfg = AutoConfig.from_pretrained(modelId, token=TOKEN)
        for key, val in configOverrides.items():
            setattr(cfg, key, val)
        tokenizer = AutoTokenizer.from_pretrained(modelId, token=TOKEN)
        model = AutoModel.from_pretrained(modelId, config=cfg)
    else:
        tokenizer = AutoTokenizer.from_pretrained(modelId, token=TOKEN)
        model = AutoModel.from_pretrained(modelId, token=TOKEN)
    model.eval()
    model.to(DEVICE)
    return model, tokenizer


def encodeTexts(modelId: str, texts: list, configOverrides: dict | None = None) -> list:
    model, tokenizer = loadEncoder(modelId, configOverrides)
    embs = []
    total = (len(texts) + BATCH - 1) // BATCH
    step = max(total // 10, 1)
    with torch.no_grad():
        for batchIdx, start in enumerate(range(0, len(texts), BATCH)):
            batch = texts[start:start + BATCH]
            if batchIdx % step == 0:
                print(f"encode {modelId}: batch {batchIdx + 1}/{total}")
            tok = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            tok = {key: val.to(DEVICE) for key, val in tok.items()}
            hidden = model(**tok).last_hidden_state
            mask = tok["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            embs.extend(pooled.cpu().tolist())
    return embs


def measureClassifyMs(modelId: str, texts: list, sample: int = 100, configOverrides: dict | None = None) -> float:
    model, tokenizer = loadEncoder(modelId, configOverrides)
    subset = texts[:max(min(sample, len(texts)), 1)]
    durs = []
    with torch.no_grad():
        for text in subset:
            tok = tokenizer([text], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            tok = {key: val.to(DEVICE) for key, val in tok.items()}
            start = time.perf_counter()
            hidden = model(**tok).last_hidden_state
            mask = tok["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            pooled.cpu().tolist()
            durs.append((time.perf_counter() - start) * 1000)
    return round(sum(durs) / max(len(durs), 1), 2)


def binaryProba(trainFeats: torch.Tensor, trainBinary: torch.Tensor, testFeats: torch.Tensor) -> torch.Tensor:
    clf = FlashLogisticRegression(C=PROBE_C)
    clf.fit(trainFeats, trainBinary)
    return torch.as_tensor(clf.predict_proba(testFeats), device=DEVICE)[:, 1]


def macroF1(pred: torch.Tensor, gold: torch.Tensor) -> float:
    scores = []
    for cls in range(N_CLASSES):
        tp = ((pred == cls) & (gold == cls)).sum().item()
        fp = ((pred == cls) & (gold != cls)).sum().item()
        fn = ((pred != cls) & (gold == cls)).sum().item()
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return sum(scores) / len(scores)


def scoreEmbeddings(embs: list, labels: list) -> dict:
    feats = torch.tensor(embs, device=DEVICE)
    gold = torch.tensor(labels, device=DEVICE)
    gen = torch.Generator().manual_seed(SEED)
    trainRows = []
    testRows = []
    for cls in range(N_CLASSES):
        idx = (gold == cls).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx), generator=gen)].tolist()
        cut = int(len(perm) * TRAIN_FRAC)
        trainRows.extend(perm[:cut])
        testRows.extend(perm[cut:])
    trainRows = [trainRows[pos] for pos in torch.randperm(len(trainRows), generator=gen).tolist()]
    testRows = [testRows[pos] for pos in torch.randperm(len(testRows), generator=gen).tolist()]
    trainIdx = torch.tensor(trainRows, device=DEVICE)
    testIdx = torch.tensor(testRows, device=DEVICE)

    torch.manual_seed(SEED)
    mean = feats[trainIdx].mean(0, keepdim=True)
    std = feats[trainIdx].std(0, keepdim=True).clamp_min(1e-6)
    trainFeats = (feats[trainIdx] - mean) / std
    testFeats = (feats[testIdx] - mean) / std
    trainLabels = gold[trainIdx]
    testLabels = gold[testIdx]
    probs = torch.zeros(len(testIdx), N_CLASSES, device=DEVICE)
    for cls in range(N_CLASSES):
        probs[:, cls] = binaryProba(trainFeats, (trainLabels == cls).long(), testFeats)
    pred = probs.argmax(1)
    return {
        "acc": (pred == testLabels).float().mean().item(),
        "f1": macroF1(pred, testLabels),
    }


def scoreWithGenerate(generate, texts: list, labels: list) -> dict:
    total = len(texts)
    correct = 0
    preds = []
    unparsed = 0
    durs = []
    for idx, (text, gold) in enumerate(zip(texts, labels)):
        start = time.perf_counter()
        completion = generate(ZS_PROMPT.format(text=text))
        durs.append((time.perf_counter() - start) * 1000)
        hit = ZS_PATTERN.search(completion)
        if hit:
            pred = ZS_LABEL_MAP[hit.group(0).lower()]
        else:
            pred = -1
            unparsed += 1
        preds.append(pred if pred >= 0 else (1 - gold if gold != 1 else 0))
        if pred == gold:
            correct += 1
        done = idx + 1
        if done % 500 == 0 or done == total:
            print(f"zs {done}/{total}")
    predTensor = torch.tensor(preds, device=DEVICE)
    goldTensor = torch.tensor(labels, device=DEVICE)
    genMs = sum(durs) / max(len(durs), 1)
    return {
        "acc": correct / total if total else 0.0,
        "f1": macroF1(predTensor, goldTensor),
        "unparsed": unparsed,
        "latencyMs": {"genPerText": round(genMs, 2),},
    }


def scoreModel(model, csvPath: str = None, mode: str = "frozen", configOverrides: dict | None = None) -> dict:
    path = csvPath or DEFAULT_CSV
    texts, labels = loadDataset(path)
    print(f"Loaded {len(texts)} labeled sentences from {path}")
    if mode == "zero-shot":
        result = scoreWithGenerate(model, texts, labels)
        return {"model": getattr(model, "__name__", "callable"), **result,}
    embs = encodeTexts(model, texts, configOverrides)
    probe = scoreEmbeddings(embs, labels)
    probe["latencyMs"] = {"classifyPerText": measureClassifyMs(model, texts, configOverrides=configOverrides),}
    return {"model": model, **probe,}
