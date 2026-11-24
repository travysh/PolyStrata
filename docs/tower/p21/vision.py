"""One picture through the CPU picture reader, both APIs: python /data/p21/vision.py. A red circle on white with the
text TOWER 21; the answer should name red and the text. Exit 1 on a miss."""
import base64, io, json, sys, time, urllib.request
from PIL import Image, ImageDraw

im = Image.new("RGB", (448, 448), "white")
d = ImageDraw.Draw(im)
d.ellipse((124, 60, 324, 260), fill=(220, 20, 20))
d.text((150, 330), "TOWER 21", fill="black", font_size=48)
buf = io.BytesIO(); im.save(buf, "PNG"); b64 = base64.b64encode(buf.getvalue()).decode()
Q = "What colour is the shape, and what does the text say? One short sentence."
bad = []


def post(path, body):
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080" + path, json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=600))
    return r, time.time() - t0


r, s = post("/v1/chat/completions", {"model": "strata", "max_tokens": 64, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}, {"type": "text", "text": Q}]}]})
a = r["choices"][0]["message"]["content"] or ""
r2, s2 = post("/v1/messages", {"model": "strata", "max_tokens": 64, "thinking": {"type": "disabled"}, "messages": [
    {"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                                 {"type": "text", "text": Q}]}]})
b = "".join(c.get("text", "") for c in r2["content"])
for name, t, sec, u in (("openai", a, s, r["usage"]), ("anthropic", b, s2, r2["usage"])):
    ok = "red" in t.lower() and "21" in t
    print(f"{'ok ' if ok else 'BAD'} vision {name}: {sec:.1f} s, usage {u}, {t.strip()!r}", flush=True)
    bad += [] if ok else [name]
sys.exit(1 if bad else 0)
