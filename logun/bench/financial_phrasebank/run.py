import json
import os
import urllib.request
from score import DEFAULT_CSV, loadDataset, scoreModel, scoreWithGenerate

MODELS = [
    ("modernbert-base", "answerdotai/ModernBERT-base", None),
    ("norberto-base", "Itau-Unibanco/NorBERTo-base", None),
    ("logun-base", "heitorrosa/logun-base", None),
    ("finbert-ptbr", "lucas-leme/FinBERT-PT-BR", None),
    ("deb3rta-base", "higopires/DeB3RTa-base", {"hidden_size": 384, "intermediate_size": 1536}),
]

# openai-api compatible
LLMS = [
    #("minicpm5-2b", "http://127.0.0.1:8080/v1/chat/completions"),
]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def llmGenerate(prompt: str, url: str) -> str:
    body = json.dumps({"messages": [{"role": "user", "content": prompt,}], "temperature": 0, "max_tokens": 5,}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json",})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())["choices"][0]["message"]["content"]


def saveResult(slug: str, result: dict) -> None:
    with open(os.path.join(RESULTS_DIR, f"{slug}.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"{slug}: {result}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    texts, labels = loadDataset(DEFAULT_CSV)
    for slug, modelId, overrides in MODELS:
        try:
            probe = scoreModel(modelId, mode="frozen", configOverrides=overrides)
            saveResult(slug, {"slug": slug, **probe,})
        except Exception as exc:
            saveResult(slug, {"slug": slug, "model": modelId, "error": str(exc),})
            
    for slug, url in LLMS:
        try:
            probe = scoreWithGenerate(lambda p, url=url: llmGenerate(p, url), texts, labels)
            saveResult(slug, {"slug": slug, "model": slug, **probe,})
        except Exception as exc:
            saveResult(slug, {"slug": slug, "model": slug, "error": str(exc),})
