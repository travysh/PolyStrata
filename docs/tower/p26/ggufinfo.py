#!/usr/bin/env python3
"""p26/ggufinfo.py FILE [PATTERN]: a GGUF file's metadata (short values) and its tensors whose names hold PATTERN."""
import struct, sys
T = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
f = open(sys.argv[1], "rb"); pat = sys.argv[2] if len(sys.argv) > 2 else "blk.5."
def rd(fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
def rs(): return f.read(rd("Q")).decode("utf-8", "replace")
def val(t):
    if t == 8: return rs()
    if t == 9:
        it, n = rd("I"), rd("Q")
        return [val(it) for _ in range(n)]
    return rd(T[t])
assert f.read(4) == b"GGUF"; ver = rd("I"); nt = rd("Q"); nk = rd("Q")
for _ in range(nk):
    k = rs(); v = val(rd("I"))
    if isinstance(v, list) and len(v) > 16: v = f"[{len(v)} items] {v[:6]}"
    if isinstance(v, str) and len(v) > 80: v = v[:80] + "..."
    if "tokenizer" not in k: print(k, "=", v)
TY = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",29:"IQ1_M",30:"BF16",39:"MXFP4"}
for _ in range(nt):
    name = rs(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; ty = rd("I"); off = rd("Q")
    if pat in name: print(name, TY.get(ty, ty), dims)
