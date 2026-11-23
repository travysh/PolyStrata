"""The `grammar` request field (GBNF, as KoboldCPP and llama.cpp take it) on /v1/chat/completions.

Checked here, before any token is generated, and sent to the engine as base64 on the GEN line.  The engine is the only
parser (src/poly/grammar.cpp): a grammar that does not parse comes back as its `ERR grammar: ...`.
"""
import base64
import copy

MAX_BYTES = 64 * 1024


class Checked(str):
    """A grammar the OpenAI route has validated.  Only this class goes to the engine, and a client's JSON can't make
    one, so a `grammar` field on any other route is ignored, as before."""


def asks_for_thinking(req):
    """The client asked for thinking (what a shared default does not count as)."""
    ctk = req.get("chat_template_kwargs") if isinstance(req.get("chat_template_kwargs"), dict) else {}
    reasoning = req.get("reasoning") if isinstance(req.get("reasoning"), dict) else {}
    return (req.get("enable_thinking") is True or ctk.get("enable_thinking") is True
            or req.get("reasoning_effort") not in (None, "", "none", False)
            or reasoning.get("effort") not in (None, "", "none", False))


def validate_request(req, engine, tools):
    """The request's grammar as a Checked string, or None when it has none.  ValueError (a 400) for one the engine
    can't apply or one combined with what it does not support.  `req` is the client's own request, before the
    server's shared thinking default was filled in."""
    value = req.get("grammar")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("grammar: expected a non-empty GBNF string")
    if len(value.encode("utf-8")) > MAX_BYTES:
        raise ValueError(f"grammar: at most {MAX_BYTES // 1024} KiB")
    if getattr(engine, "info", {}).get("grammar") != 1:
        raise ValueError("grammar: this engine or model does not support grammars; use a strata-poly model with a "
                         "byte-level BPE vocabulary")
    if getattr(engine, "batch", 0):
        raise ValueError("grammar: continuous batching is not supported; run without --batch")
    tool_choice = req.get("tool_choice")
    if tools or req.get("tools") or (tool_choice not in (None, "", "none")
                                     and not (isinstance(tool_choice, dict) and tool_choice.get("type") == "none")):
        raise ValueError("grammar with tools is not supported")
    if req.get("strata_mcp"):
        raise ValueError("grammar with MCP is not supported")
    if req.get("continue_final_message"):
        raise ValueError("grammar with a prefill (continue_final_message) is not supported")
    if asks_for_thinking(req):
        raise ValueError("grammar with thinking is not supported; the grammar applies to the answer itself")
    response_format = req.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") not in (None, "text"):
        raise ValueError("grammar with a response_format other than text is not supported")
    return Checked(value)


def engine_key(value):
    """` grammar=<base64>` for a Checked grammar, else nothing (the GEN line is then what it was)."""
    if not isinstance(value, Checked):
        return ""
    return " grammar=" + base64.b64encode(value.encode("utf-8")).decode("ascii")


def from_schema(schema):
    """The grammar llama.cpp builds for a JSON schema (serve/gbnf_schema.py), as a Checked string.  None when the
    converter can't handle the schema: the request is then prompted and checked, as with the option off."""
    from serve import gbnf_schema
    try:
        converter = gbnf_schema.SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
        resolved = converter.resolve_refs(copy.deepcopy(schema), "schema")
        converter.visit(resolved, "")
        return Checked(converter.format_grammar())
    except (ValueError, AssertionError, KeyError, TypeError, RecursionError):
        return None


def for_format(response_format, engine, kw):
    """With the server option on: the grammar for a `json_object` / `json_schema` response_format, or None where it
    can't be applied (an engine without grammars, --batch, thinking on, a schema the converter can't take).  The
    caller keeps the prompt directive and the validator either way."""
    if getattr(engine, "info", {}).get("grammar") != 1 or getattr(engine, "batch", 0):
        return None
    if kw.get("enable_thinking", True) is not False:
        return None
    kind = response_format.get("type") if isinstance(response_format, dict) else None
    if kind == "json_object":
        schema = {"type": "object"}
    elif kind == "json_schema":
        schema = response_format["json_schema"]["schema"]       # prepare_format has checked the shape
    else:
        return None
    return from_schema(schema)
