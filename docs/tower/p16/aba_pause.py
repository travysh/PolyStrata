"""aba.py with a pause before A2, so that B's park has finished writing before A comes back:
cd /opt/strata && .venv/bin/python /data/aba_pause.py LABEL SECONDS [size ...]"""
import sys, time
sys.path.insert(0, "/data")
import aba

_ask, n = aba.ask, [0]


def ask(messages):
    n[0] += 1
    if n[0] % 4 == 3:
        time.sleep(float(sys.argv[2]))
    return _ask(messages)


aba.ask = ask
for s in sys.argv[3:] or ["24000"]:
    aba.run(sys.argv[1], int(s))
