import json
import subprocess
import urllib.request

PROMPT = (
    "O Banco Central manteve a taxa Selic em dois digitos e o mercado de juros futuros "
    "reagiu com alta nos vencimentos longos enquanto a bolsa recuou com bancos e varejo "
    "liderando as perdas do dia diante do pessimismo externo sobre inflacao e credito."
)
URL = "http://127.0.0.1:8080/completion"


def post_once():
    body = json.dumps({"prompt": PROMPT, "n_predict": 128, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def vram_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return int(out.stdout.strip().split("\n")[0].strip())
    except Exception:
        return -1


# ponytail: stdlib urllib only, no requests dep.
post_once()  # run 1: first-touch warmup, discarded
r2 = post_once()  # run 2: measured

text = r2.get("content", "")
t = r2.get("timings", {}) or r2.get("timings", {})
n, ms = t.get("predicted_n", 0), t.get("predicted_ms", 0)
toks = (n / (ms / 1000.0)) if n and ms else 0.0
sane = bool(text) and len(text) > 50
print(json.dumps({"toksPerSec": round(toks, 2), "vramMiB": vram_mib(), "sane": sane}))
