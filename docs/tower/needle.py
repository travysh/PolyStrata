"""Long-context recall check: three facts hidden at 3%, 50% and 92% depth of a ~105K-token prompt.
Run inside the container: cd /opt/strata && .venv/bin/python /data/needle.py [seed ...]"""
import json, random, sys, time, urllib.request
sys.path.insert(0, "/opt/strata")
from polystrata import long_prompt

BASE = "http://127.0.0.1:8080"
WORDS = ["amber", "cobalt", "falcon", "juniper", "marble", "nickel", "orchid", "quartz", "saffron", "tundra"]


def trial(seed: int) -> bool:
    rng = random.Random(seed)
    facts = [(f"{rng.choice(WORDS)}-{rng.randrange(1000, 9999)}", label)
             for label in ("the vault on the first floor", "the vault on the second floor", "the vault on the roof")]
    body = long_prompt(80000)
    body = body[body.index("\n\n") + 2:body.rindex("\n\nIn three")]          # the manual text alone
    for (code, label), depth in sorted(zip(facts, (0.03, 0.50, 0.92)), key=lambda x: -x[1]):
        at = body.index("\n", int(len(body) * depth)) + 1                     # on a line of its own
        body = body[:at] + f"IMPORTANT NOTE: the access code for {label} is {code}.\n" + body[at:]
    q = ("Here is part of a manual:\n\n" + body + "\n\nThe manual above contains three IMPORTANT NOTEs with access "
         "codes. List the access code for each of the three vaults (first floor, second floor, roof).")
    req = {"model": "strata", "messages": [{"role": "user", "content": q}], "max_tokens": 4096, "temperature": 0}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=7200))
    msg = r["choices"][0]["message"]
    answer = msg.get("content") or ""
    hits = [code in answer for code, _ in facts]
    print(f"seed {seed}: {r['usage']['prompt_tokens']} prompt tokens, {time.time() - t0:.0f} s, "
          f"facts {[c for c, _ in facts]} -> found (3%, 50%, 92%) {hits}")
    print("  answer:", answer.strip().replace("\n", " | ")[:400])
    return all(hits)


if __name__ == "__main__":
    ok = [trial(int(s)) for s in (sys.argv[1:] or ["1", "2"])]
    print("PASS" if all(ok) else "FAIL")
    sys.exit(0 if all(ok) else 1)
