"""p26/encode.py TOKENIZER_DIR OUT_DIR: the six chats of p26/chats.py as the server encodes them (its tokenizer, its
chat template, thinking off), one file of comma-separated token ids each (OUT_DIR/chatN.txt), for strata-poly
--tokens.  Run from /opt/strata/terminal."""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import serve.server as S
sys.path.insert(0, "/p26")
from chats import P

tpath, out = Path(sys.argv[1]), Path(sys.argv[2])
tok = S.load_tokenizer(tpath)
tpl = tpath / "chat_template.jinja"
svc = S.Service(S.MockEngine(tok, ["x"]), tok, S.ChatTemplate(tpl if tpl.exists() else S.ROOT / "serve/chat_template.jinja"))
for i, p in enumerate(P):
    body = {"messages": [{"role": "user", "content": p}], "chat_template_kwargs": {"enable_thinking": False}}
    messages, tools, kwargs = S.openai_to_messages(body)
    ids = svc.encode_prompt(messages, tools, kwargs, store=False)
    (out / f"chat{i}.txt").write_text(",".join(map(str, ids)))
    print(f"chat{i}: {len(ids)} tokens")
