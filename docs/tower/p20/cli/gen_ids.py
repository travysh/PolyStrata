"""Token ids for the engine's one-shot mode: short.ids and long.ids (11K), Qwen chat form, thinking off."""
import sys
sys.path.insert(0, "/opt/strata")
sys.path.insert(0, "/opt/strata/terminal/tools")
import strata_tokenizer as ST
from polystrata import long_prompt

tok = ST.Tokenizer.from_dir("/data/packs/cybertiel-ud-q4_k_xl/tokenizer")
Q = "Write a Python function that parses an INI file into a dict of dicts, with comments and tests."


def chat(text):
    s = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return tok.encode(s, parse_special=True)


for name, text in (("short", Q), ("long", f"Session 1.\n\n{long_prompt(8000)}\n\n{Q}")):
    ids = chat(text)
    open(f"/data/p20cli/{name}.ids", "w").write(",".join(map(str, ids)))
    print(name, len(ids), len(",".join(map(str, ids))), "bytes")
