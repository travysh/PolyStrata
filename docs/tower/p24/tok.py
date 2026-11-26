import sys; sys.path.insert(0, "serve")
from pathlib import Path
import server
t = server.load_tokenizer(Path("/data/packs/inkling-ud-q4_k_xl/tokenizer"))
msg = [{"role": "user", "content": "Write a long, detailed story about a lighthouse keeper who finds a strange machine on the beach."}]
try:
    ids = t.apply_chat_template(msg, add_generation_prompt=True, tokenize=True)
except Exception as e:
    print("no template:", e, file=sys.stderr); ids = t.encode(msg[0]["content"])
if hasattr(ids, "ids"): ids = ids.ids
if isinstance(ids, dict): ids = ids["input_ids"]
print(",".join(map(str, ids)))
