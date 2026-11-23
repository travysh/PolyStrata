"""The add-on generator's pattern: a long shared "forge" system prompt with a new user turn each time, alternating
with short requests under other system prompts (texture, refine).  Per forge request: tokens taken from the cache
and prompt_ms.  Inside the container: python /data/addon_sim.py LABEL [rounds]"""
import json, random, sys, time, urllib.request
from polystrata import long_prompt

BASE = "http://127.0.0.1:8080"
FORGE_SYS = "You are the forge of a Minecraft add-on generator.\n\n" + long_prompt(10000)
OTHER_SYS = ["You draw 16x16 pixel art textures as JSON grids.\n\n" + long_prompt(600, 0.5),
             "You refine Minecraft mob designs.\n\n" + long_prompt(1500, 0.8)]


def ask(system, user, max_tokens=16):
    req = {"model": "strata", "max_tokens": max_tokens, "temperature": 0,
           "chat_template_kwargs": {"enable_thinking": False},
           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=3600))
    t = r.get("timings") or {}
    return r["usage"]["prompt_tokens"], t.get("cache_n") or 0, t.get("prompt_ms") or 0.0, time.time() - t0


label, rounds = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8
rng = random.Random(7)
tot_p = tot_c = 0
ms = []
for i in range(rounds):
    user = f"Request {i}: make a {rng.choice(['mob', 'item', 'block'])} called thing-{rng.randrange(10**6)}.\n\n" + \
           long_prompt(1300, rng.uniform(0.2, 1.0))
    p, c, pm, w = ask(FORGE_SYS, user)
    tot_p += p
    tot_c += c
    if i:
        ms.append(pm)
    print(f"{label} forge {i}: prompt={p} cached={c} prompt_ms={pm:.0f} wall_s={w:.1f}", flush=True)
    for s in OTHER_SYS:
        p2, c2, pm2, _ = ask(s, f"Item {i} {rng.randrange(10**6)}: describe it.")
        print(f"{label}   other: prompt={p2} cached={c2} prompt_ms={pm2:.0f}", flush=True)
print(f"{label} forge reuse {tot_c}/{tot_p} = {100 * tot_c / tot_p:.1f}%, prompt_ms after the first: "
      f"median {sorted(ms)[len(ms) // 2]:.0f}, max {max(ms):.0f}", flush=True)
