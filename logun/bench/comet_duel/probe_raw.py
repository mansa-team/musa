import json
import sys
import urllib.request

sys.path.insert(0, "logun/bench/comet_duel")
from translate import SYSTEM, EXAMPLES

samples = json.load(open("logun/bench/comet_duel/smoke5.json", encoding="utf-8"))
src = samples[0].get("text", samples[0].get("en", ""))
body = json.dumps({
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Examples:\n" + EXAMPLES + "\n\n<source>" + src + "</source>"},
        {"role": "assistant", "content": "<translation>"},
    ],
    "temperature": 0,
    "n_predict": 128,
    "stream": False,
    "stop": ["</translation>", "\n\n\n"],
}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
resp = json.loads(urllib.request.urlopen(req, timeout=180).read().decode())
msg = resp["choices"][0]["message"]
print("KEYS:", list(msg.keys()))
for key, val in msg.items():
    print(key.upper(), ":", repr(val)[:800])
