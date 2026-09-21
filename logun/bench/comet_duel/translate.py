import argparse
import json
import urllib.request

SYSTEM = ("You are a professional translator specialized in financial news. "
          "Translate the English sentence inside <source> tags into Brazilian "
          "Portuguese (pt-BR). Rules: 1. Output ONLY the translation - no "
          "explanations, no quotation marks, no thinking, no preamble. "
          "2. Preserve numbers, percentages, currency codes, tickers and "
          "proper names exactly (e.g. EUR3.1m, 42.5%, AFX). "
          "3. Use Brazilian financial terminology. "
          "4. Never output the English source.")
EXAMPLES = ("""<source>Net profit rose to EUR3.1 million from EUR2.5 million.</source>
<translation>O lucro líquido subiu para 3,1 milhões de euros, ante 2,5 milhões de euros.</translation>

<source>Operating profit increased by 42.5% from 2004.</source>
<translation>O lucro operacional aumentou 42,5% em relação a 2004.</translation>
""")
TIMEOUT = 180
PREFILL = 'A tradução para o português brasileiro é: "'
THINK_MARKERS = ("The user wants", "We need to", "<think>", "</think>")


def strip_prefill(mt: str) -> str:
    s = mt.strip()
    if s.startswith(PREFILL):
        s = s[len(PREFILL):]
    s = s.lstrip('"')
    return s.strip()


def translateOne(text: str, url: str) -> str:
    body = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": (
                "<source>" + text + "</source>\n\nExamples:\n"
                + EXAMPLES
                + "\n<source>" + text + "</source>\n<translation>")},
            {"role": "assistant", "content": PREFILL},
        ],
        "temperature": 0,
        "n_predict": max(64, 3 * len(text.split())),
        "stream": False,
        "stop": ["\n\n"],
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        msg = json.loads(resp.read().decode())["choices"][0]["message"]
        content = msg.get("content") or ""
        if not content.strip() and msg.get("reasoning_content"):
            content = msg["reasoning_content"]
        elif msg.get("reasoning_content") and content.strip() == msg["reasoning_content"].strip():
            pass
        return content


def main(argv: list = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args(argv)
    with open(args.inp, "r", encoding="utf-8") as handle:
        samples = json.load(handle)
    out = []
    for row in samples:
        src = row.get("text", row.get("en", ""))
        try:
            mt = strip_prefill(translateOne(src, args.url))
            out.append({"id": row["id"], "en": src, "src": src, "mt": mt,
                        "ref": row.get("ref"), "y": row.get("y"),
                        "model": args.model_name})
        except Exception as exc:
            out.append({"id": row["id"], "en": src, "src": src, "mt": "",
                        "ref": row.get("ref"), "y": row.get("y"),
                        "model": args.model_name, "error": str(exc)})
        done = len(out)
        if done % 5 == 0 or done == len(samples):
            print(f"{args.model_name} {done}/{len(samples)}", flush=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)
    empty = sum(1 for r in out if not r["mt"].strip())
    resid = sum(1 for r in out if any(m.lower() in r["mt"].lower() for m in THINK_MARKERS)
                or (r["src"] and r["src"][:30] in r["mt"]))
    print(f"wrote {len(out)} translations to {args.out} empty={empty} residue={resid}")


if __name__ == "__main__":
    main()
