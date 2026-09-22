import argparse
import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

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

## Fine-Tuning & In-Context Demonstrations
"""
EXAMPLES = """
<source>Orion Pharma's net profit rose to EUR3.1m from EUR2.5m in Q2 2010.</source>
<translation>O lucro líquido da Orion Pharma subiu para EUR 3,1m, ante EUR 2,5m no 2º trimestre de 2010.</translation>

<source>It generated an operating loss of EUR 96.3 mn, down from a profit of EUR 43.9 mn.</source>
<translation>A empresa gerou um prejuízo operacional de EUR 96,3 mi, em comparação ao lucro de EUR 43,9 mi registrado anteriormente.</translation>
"""

LOOSE_SYS = "You are a translator. Translate the English financial news text into Brazilian Portuguese. Output only the translation, no explanations, no quotes."
PREFILL_TAG = "<translation>"
TIMEOUT = 180
PREFILL = 'A tradução para o português brasileiro é: "'
THINK_MARKERS = ("The user wants", "We need to", "<think>", "</think>")


def strip_prefill(mt: str) -> str:
    s = mt.strip()
    if s.startswith(PREFILL):
        s = s[len(PREFILL):]
    s = s.lstrip('"')
    return s.strip()


def translateOne(text: str, url: str, style: str = "full", temperature: float = 0.0) -> str:
    if style == "loose":
        messages = [
            {"role": "system", "content": LOOSE_SYS},
            {"role": "user", "content": text},
            {"role": "assistant", "content": PREFILL_TAG},
        ]
    elif style == "noex":
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<source>{text}</source>"},
            {"role": "assistant", "content": PREFILL_TAG},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Examples:\n{EXAMPLES}\n\n<source>{text}</source>"},
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


def main(argv: list = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prompt-style", choices=["full", "noex", "loose"], default="full")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)
    with open(args.inp, "r", encoding="utf-8") as handle:
        samples = json.load(handle)
    out = [None] * len(samples)
    done = 0
    lock = threading.Lock()

    def work(idx: int) -> None:
        nonlocal done
        row = samples[idx]
        src = row.get("text", row.get("en", ""))
        try:
            mt = strip_prefill(translateOne(src, args.url, args.prompt_style, args.temperature))
            out[idx] = {"id": row["id"], "en": src, "src": src, "mt": mt,
                        "ref": row.get("ref"), "y": row.get("y"),
                        "model": args.model_name}
        except Exception as exc:
            out[idx] = {"id": row["id"], "en": src, "src": src, "mt": "",
                        "ref": row.get("ref"), "y": row.get("y"),
                        "model": args.model_name, "error": str(exc)}
        with lock:
            done += 1
            if done % 5 == 0 or done == len(samples):
                print(f"{args.model_name} {done}/{len(samples)}", flush=True)

    workers = max(1, args.workers)
    if workers == 1:
        for idx in range(len(samples)):
            work(idx)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, range(len(samples))))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)
    empty = sum(1 for r in out if not r["mt"].strip())
    resid = sum(1 for r in out if any(m.lower() in r["mt"].lower() for m in THINK_MARKERS)
                or (r["src"] and r["src"][:30] in r["mt"]))
    print(f"wrote {len(out)} translations to {args.out} empty={empty} residue={resid}")


if __name__ == "__main__":
    main()
