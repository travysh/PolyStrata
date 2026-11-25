"""Send the Pass 22 tasks to the model the server runs and keep the answers (graded later, apart, by grade.py).

Run inside the model's container (it talks to the server on 127.0.0.1:8080):
    python run_bench.py TASKS OUT.jsonl SAMPLES [ids...]
Both models get the same request: the task's prompt as the only user message, thinking on, the same sampling and the
same token limits, a seed per sample.  A finished (id, sample) in OUT.jsonl is not asked again, so a run can resume."""
import json, sys, time, urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8080"
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
MAX_TOKENS = 16384                  # thinking and answer together
THINK_BUDGET = 12288                # thinking stops here, so an answer always has room


def ask(prompt, seed, extra=None):
    body = {"model": "strata", "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS,
            "reasoning_budget_tokens": THINK_BUDGET, "seed": seed, "stream": False, **SAMPLING,
            **(extra or {})}          # a task's own settings (think.py: thinking off, budgets)
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    msg, usage = r["choices"][0]["message"], r.get("usage", {})
    return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
            "finish": r["choices"][0].get("finish_reason"), "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "wall_s": round(time.time() - t0, 1)}


def main():
    tasks, out, samples, want = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), set(sys.argv[4:])
    done = set()
    if out.exists():
        done = {(r["id"], r["sample"]) for r in map(json.loads, out.read_text().splitlines())}
    model = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]
    todo = [json.loads((d / "task.json").read_text()) for d in sorted(tasks.iterdir()) if (d / "task.json").exists()]
    todo = [t for t in todo if not want or t["id"] in want]
    for s in range(samples):                         # sample by sample, so a cut-short run still covers every task
        for t in todo:
            if (t["id"], s) in done:
                continue
            try:
                r = ask(t["prompt"], seed=1000 + s, extra=t.get("request"))
            except Exception as e:                   # a failed request is an answer of its own: it scores 0
                r = {"content": "", "reasoning": "", "finish": "error: " + str(e)[:200]}
            r.update(id=t["id"], kind=t["kind"], sample=s, model=model)
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{t['id']} s{s} {r['finish']} out={r.get('completion_tokens')} think={len(r['reasoning'])}ch "
                  f"answer={len(r['content'])}ch {r.get('wall_s')}s", flush=True)


if __name__ == "__main__":
    main()
