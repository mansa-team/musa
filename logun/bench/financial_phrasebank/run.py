import json
import os
import subprocess
import sys
import time
import urllib.request

import yaml
from pathlib import Path
from score import DEFAULT_CSV, loadDataset, scoreModel, scoreWithGenerate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGUN_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))


CONFIG = yaml.safe_load((Path(__file__).resolve().parent.parent.parent / "config.yaml").read_text(encoding="utf-8"))


URL = f"http://{CONFIG['llm']['host']}:{CONFIG['llm']['port']}"
MODELS_FILE = os.path.join(LOGUN_DIR, "models.json")
LAUNCH = os.path.join(LOGUN_DIR, "inference", "launch.py")

SKIP_LLMS = False

MODELS = [
    ("modernbert-base", "answerdotai/ModernBERT-base", None),
    ("norberto-base", "Itau-Unibanco/NorBERTo-base", None),
    ("logun-base", "heitorrosa/logun-base", None),
    ("finbert-ptbr", "lucas-leme/FinBERT-PT-BR", None),
    ("deb3rta-base", "higopires/DeB3RTa-base", {"hidden_size": 384, "intermediate_size": 1536}),
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def llmGenerate(prompt: str, url: str) -> str:
    body = json.dumps({"messages": [{"role": "user", "content": prompt,}], "temperature": 0, "max_tokens": 5,}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json",})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())["choices"][0]["message"]["content"]


def wait_healthy(timeout: int = 180) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise RuntimeError("server never became healthy")


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

    with open(MODELS_FILE, "r", encoding="utf-8") as handle:
        engines = [] if SKIP_LLMS else json.load(handle)
    endpoint = URL + "/v1/chat/completions"
    for m in engines:
        slug = m["name"]
        try:
            print("+ launch " + slug, flush=True)
            subprocess.run([sys.executable, LAUNCH, "--model", os.path.normpath(os.path.join(LOGUN_DIR, m["gguf"]))] + m.get("flags", []), check=True)
            try:
                wait_healthy()
                probe = scoreWithGenerate(lambda p, url=endpoint: llmGenerate(p, url), texts, labels)
                saveResult(slug, {"slug": slug, "model": slug, **probe,})
            finally:
                subprocess.run([sys.executable, LAUNCH, "stop"], check=True)
        except Exception as exc:
            saveResult(slug, {"slug": slug, "model": slug, "error": str(exc),})
