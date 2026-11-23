"""Compare the token texts strata-poly's grammar matches with the server's tokenizer, for every id.

    python tools/test_poly_vocab_texts.py MODEL.gguf [path/to/grammar_test]

`grammar_test --dump` (src/poly/grammar_test.cpp) writes "id hex" for each token, decoded from the GGUF's byte-level
BPE pieces by src/poly/vocab_text.hpp.  tools/strata_tokenizer.py decodes the same pieces for the server
(`token_bytes`).  They must agree on every id except a control token, which the grammar gives no text.
Exit status 1 and the first differences when they do not.
"""
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import strata_tokenizer  # noqa: E402


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    model, tool = argv[1], argv[2] if len(argv) > 2 else "./grammar_test"
    tok = strata_tokenizer.Tokenizer.from_gguf(model)
    dump = subprocess.run([tool, "--dump", model], capture_output=True, text=True)
    if dump.returncode:
        print("grammar_test --dump failed:", dump.stderr.strip())
        return 1
    texts = {}
    for line in dump.stdout.splitlines():
        i, _, hexed = line.partition(" ")
        texts[int(i)] = bytes.fromhex(hexed)
    types = tok.token_types or []
    bad, control = [], 0
    for i in range(len(tok.tokens)):
        if i < len(types) and types[i] == 3:
            control += 1
            if texts.get(i):
                bad.append((i, texts[i], b""))
            continue
        want = tok.token_bytes(i)
        if texts.get(i, b"") != want:
            bad.append((i, texts.get(i, b""), want))
    print(f"{len(tok.tokens)} tokens, {control} control, {len(bad)} differ")
    for i, got, want in bad[:10]:
        print(f"  id {i} {tok.tokens[i]!r}: engine {got!r}, server {want!r}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
