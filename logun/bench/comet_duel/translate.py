import json
import os
import sys
import threading
import urllib.request

import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load((Path(__file__).resolve().parent.parent.parent / "config.yaml").read_text(encoding="utf-8"))

URL = f"http://{CONFIG['llm']['host']}:{CONFIG['llm']['port']}"

SYSTEM = """
Current system role: High-Throughput Neural Machine Translation & Financial NLP Curation Engine

You are a specialized Neural Machine Translation (NMT) and Financial NLP Curation Engine built to prepare synthetic financial parallel corpora for domain-adaptive pre-training (DAPT) and supervised fine-tuning (SFT) of Portuguese language models. You operate with absolute semantic fidelity, zero information loss, and strict compliance with Brazilian capital markets terminology (B3, CVM, ANBIMA).

## Mission & Core Curation Principles

- ABSOLUTE ENTITY PRESERVATION: Named entities (corporations, executives, regulatory bodies, geographic regions, ticker symbols) are sacred. Omitting, truncating, or summarizing an entity destroys downstream token-alignment for classification and NER models. Every entity in <source> MUST have its exact equivalent in <translation>.
- SATELLITE CONTEXT INTEGRITY: Temporal qualifiers ("Q3 2024", "as of mid-September", "YoY"), percentage points, basis points, and currency values carry directional weight. Stripping timeframe qualifiers is treated as a critical execution failure.
- SEMANTIC DIRECTIONALITY: Financial sentiment is fragile. Never flip directional sentiment. "Prejuízo" (loss) and "Lucro" (profit) are non-interchangeable; mistranslating directional movement ("rose", "fell", "stagnated") corrupts downstream classification data.
- TERMINOLOGICAL RIGOR: Use institutional Brazilian Portuguese capital markets jargon (e.g., use "prejuízo operacional" instead of "perda operacional"; "oferta pública inicial" instead of "primeira venda de ações"; "reagrupamento/inplit" for reverse splits).
- ZERO PREAMBLE / ZERO METADATA: You operate in an automated data pipeline. Output ONLY the translated target text enclosed inside <translation> tags. Never output conversational responses, explanations, reasoning tags (<think>), or markdown formatting outside the designated tags.

## Operational Execution Guidelines

1. EXACT FORMATTING PRESERVATION:
   - Currency formats: Preserve monetary units (e.g., "EUR3.1m" -> "EUR 3,1m" or "3,1 milhões de euros"; "R$ 50M" -> "R$ 50M").
   - Percentages & Decimals: Convert English decimal dots to Brazilian decimal commas where appropriate (e.g., "42.5%" -> "42,5%").
   - Tickers & Proper Nouns: Keep original corporate names and tickers intact (e.g., "SanomaWSOY", "AFX", "PETR4"). Never attempt to translate proper corporate names into Portuguese unless an official localized brand exists.

2. IDIOM & STYLISTIC ADAPTATION:
   - Translate financial idioms by semantic intent, not verbatim literalism.
   - Example: "putting a stake in the ground" -> "definindo um posicionamento/marco firme" (NOT "colocando uma estaca no chão" or "participação nos trinta").
   - Example: "fell short of expectations" -> "ficou aquém das expectativas" or "veio abaixo do consenso".

3. HARD CONSTRAINTS & FAILURE MODES TO AVOID:
   - DO NOT truncate long trailing clauses or multi-sentence paragraphs.
   - DO NOT drop company names even if they appear redundant in the prompt.
   - DO NOT add quotation marks around the translated output unless present in the source text.
   - DO NOT echo back the English source text.

## Structural Input/Output Contract

Input will always be supplied inside <source> tags.
Output MUST strictly follow this exact syntax structure:

<translation>
[Translated text in flawless Brazilian Portuguese financial terminology]
</translation>
"""
PREFILL_TAG = "<translation>"
TIMEOUT = 180
THINK_MARKERS = ("The user wants", "We need to", "<think>", "</think>")


def translateOne(text: str, url: str, temperature: float = 0.0) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"<source>{text}</source>"},
        {"role": "assistant", "content": PREFILL_TAG},
    ]
    body = json.dumps({
        "messages": messages,
        "temperature": temperature,
        "n_predict": max(128, 4 * len(text.split())),
        "stream": False,
        "stop": ["</translation>", "\n\n\n"],
    }).encode()

    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        msg = json.loads(resp.read().decode())["choices"][0]["message"]
        content = msg.get("content") or ""

        content = content.replace("<translation>", "").replace("</translation>", "").strip()
        return content


def run(cfg: dict) -> None:
    url, inp = cfg.get("url") or URL, cfg["samples"]
    model_name = cfg["model_name"]
    out_path = cfg.get("out") or os.path.join(SCRIPT_DIR, "results", model_name + ".json")
    workers = max(1, cfg.get("workers", 4))
    temperature = cfg.get("temperature", 0.0)
    with open(inp, "r", encoding="utf-8") as handle:
        samples = json.load(handle)
    out = [None] * len(samples)
    done = 0
    lock = threading.Lock()

    def work(idx: int) -> None:
        nonlocal done
        row = samples[idx]
        src = row.get("text", row.get("en", ""))
        try:
            mt = translateOne(src, url, temperature).strip()
            out[idx] = {"id": row["id"], "en": src, "src": src, "mt": mt,
                        "ref": row.get("ref"), "y": row.get("y"),
                        "model": model_name}
        except Exception as exc:
            out[idx] = {"id": row["id"], "en": src, "src": src, "mt": "",
                        "ref": row.get("ref"), "y": row.get("y"),
                        "model": model_name, "error": str(exc)}
        with lock:
            done += 1
            if done % 5 == 0 or done == len(samples):
                print(f"{model_name} {done}/{len(samples)}", flush=True)

    if workers == 1:
        for idx in range(len(samples)):
            work(idx)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, range(len(samples))))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)
    empty = sum(1 for r in out if not r["mt"].strip())
    resid = sum(1 for r in out if any(m.lower() in r["mt"].lower() for m in THINK_MARKERS)
                or (r["src"] and r["src"][:30] in r["mt"]))
    print(f"wrote {len(out)} translations to {out_path} empty={empty} residue={resid}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="")
    parser.add_argument("--samples", default=os.path.join(SCRIPT_DIR, "samples.json"))
    parser.add_argument("--out", default="")
    parser.add_argument("--model-name", default="lfm")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    run({"url": args.url, "samples": args.samples, "out": args.out or None,
         "model_name": args.model_name, "workers": args.workers,
         "temperature": args.temperature})
