"""aba.py with a fixed nonce, printing each answer, so two engines can be compared answer for answer:
cd /opt/strata && .venv/bin/python /data/aba_eq.py LABEL [size ...]"""
import hashlib, random, sys
sys.path.insert(0, "/data")
import aba

_ask = aba.ask


def ask(messages):
    r, wall, t = _ask(messages)
    c = r["choices"][0]["message"].get("content") or ""
    print(f"  answer sha1={hashlib.sha1(c.encode()).hexdigest()[:12]} {c[:80]!r}", flush=True)
    return r, wall, t


aba.ask = ask
random.seed(1)
for s in sys.argv[2:] or ["24000"]:
    aba.run(sys.argv[1], int(s))
