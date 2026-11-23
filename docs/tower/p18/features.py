"""The tower-only API features against a running server: python /data/features.py [base_url]
grammar field, constrained response_format json_schema, assistant prefill (both APIs), logit_bias. Exit 1 on a miss."""
import json, sys, urllib.error, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
bad = []


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return 200, json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def check(name, ok, detail):
    print(f"{'ok ' if ok else 'BAD'} {name}: {detail}", flush=True)
    if not ok:
        bad.append(name)


health = json.load(urllib.request.urlopen(URL + "/v1/status", timeout=30))
so = health.get("structured_output", {})
check("health", so.get("constrained_decoding") is True, so)

chat = {"model": "strata", "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
# grammar: only yes or no
code, r = post("/v1/chat/completions", {**chat, "messages": [{"role": "user", "content": "Is the sky green?"}],
                                       "grammar": 'root ::= "yes" | "no"'})
text = r.get("choices", [{}])[0].get("message", {}).get("content") if code == 200 else r
check("grammar", code == 200 and text in ("yes", "no"), repr(text))

# constrained json_schema
schema = {"type": "object", "properties": {"name": {"type": "string"}, "hp": {"type": "integer"}},
          "required": ["name", "hp"], "additionalProperties": False}
code, r = post("/v1/chat/completions", {**chat, "max_tokens": 200, "messages": [
    {"role": "user", "content": "Make a Minecraft mob with a name and hit points."}],
    "response_format": {"type": "json_schema", "json_schema": {"name": "mob", "schema": schema}}})
text = r.get("choices", [{}])[0].get("message", {}).get("content") if code == 200 else r
try:
    obj = json.loads(text)
    ok = set(obj) == {"name", "hp"} and isinstance(obj["hp"], int)
except Exception:
    ok = False
check("json_schema", code == 200 and ok, text)

# prefill, OpenAI (continue_final_message) and Anthropic (a last assistant message)
code, r = post("/v1/chat/completions", {**chat, "continue_final_message": True, "messages": [
    {"role": "user", "content": "Count from one to five in words."}, {"role": "assistant", "content": "one, two,"}]})
text = r.get("choices", [{}])[0].get("message", {}).get("content") if code == 200 else r
check("prefill openai", code == 200 and "three" in str(text).lower() and "one, two" not in str(text), repr(text))
code, r = post("/v1/messages", {"model": "strata", "max_tokens": 64, "messages": [
    {"role": "user", "content": "Count from one to five in words."}, {"role": "assistant", "content": "one, two,"}]})
text = "".join(b.get("text", "") for b in r.get("content", [])) if code == 200 else r
check("prefill anthropic", code == 200 and "three" in str(text).lower(), repr(text))

# logit_bias: ban the greedy first token and see the answer change
q = {**chat, "max_tokens": 8, "messages": [{"role": "user", "content": "Reply with one English greeting, one word only."}]}
code, r = post("/v1/chat/completions", q)
a = r["choices"][0]["message"]["content"] if code == 200 else r
check("logit_bias base", code == 200, repr(a))
import glob, os
sys.path.insert(0, next(d for d in ("/opt/strata/terminal/tools", "/opt/strata/tools") if os.path.isdir(d)))
import strata_tokenizer as ST
tok = ST.Tokenizer.from_dir("/data/packs/cybertiel-ud-q4_k_xl/tokenizer")
ids = tok.encode(a) if code == 200 else []
info = {}
if ids:
    code, r = post("/v1/chat/completions", {**q, "logit_bias": {str(ids[0]): -100}})
    b = r["choices"][0]["message"]["content"] if code == 200 else r
    check("logit_bias -100", code == 200 and b != a, f"{a!r} -> {b!r} (banned {ids[0]})")
else:
    check("logit_bias -100", False, "no tokens")

print("ALL OK" if not bad else f"MISSES: {bad}")
sys.exit(1 if bad else 0)
