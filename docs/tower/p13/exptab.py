"""Writes the routed experts' file offsets for cxb.cpp, one line per MoE layer (the MTP block's too, last):
layer gu_type d_type gate_off up_off down_off gu_stride d_stride (offsets absolute in the file, strides per expert).
Run in the container: PYTHONPATH=third_party/llama.cpp/gguf-py .venv/bin/python exptab.py GGUF > tab.txt"""
import sys
import gguf

r = gguf.GGUFReader(sys.argv[1])
t = {x.name: x for x in r.tensors}
for l in range(1000):
    g, u, d = (t.get(f"blk.{l}.ffn_{k}_exps.weight") for k in ("gate", "up", "down"))
    if g is None:
        continue
    n_exp = int(g.shape[-1])
    print(l, int(g.tensor_type), int(d.tensor_type), g.data_offset, u.data_offset, d.data_offset,
          g.n_bytes // n_exp, d.n_bytes // n_exp)
