"""Inkling 2 behind the server: its pre-tokenizer, the typed blocks its replies are made of, and a reply that must
not think starting in its text block.

    python -m unittest serve.test_inkling

The golden cases (serve/fixtures/inkling2_golden.json): the token ids of 51 strings (53 cases) from Hugging Face's
tokenizers on thinkingmachines/Inkling-2-Mini's tokenizer.json (the reference; llama.cpp's where they differ are kept
as "llama_cpp_ids"; the cases with control tokens read as text are llama.cpp's), checked with the part of the
vocabulary they use (serve/fixtures/inkling2_vocab_subset.json) and with the whole vocabulary from
$STRATA_INKLING_TOKENIZER (a pack's tokenizer/ directory) when it is set; llama.cpp 29b52d398b's rendered prompts for
OpenAI requests with the model's template (serve/fixtures/family_templates.json.gz "inkling"); and llama.cpp's parse
of replies.
"""
import gzip
import hashlib
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST  # noqa: E402
from serve.frontend import (FUNC_START, INK_TEXT_OPEN, ChatTemplate, InklingOutputParser,  # noqa: E402
                            anthropic_to_messages, forced_call, inkling_forced_call, openai_to_messages)
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN = json.loads((FIXTURES / "inkling2_golden.json").read_text(encoding="utf-8"))
# the model's own template (the GGUF's tokenizer.chat_template, thinkingmachines/Inkling-2-Mini's chat_template.jinja)
MODEL_TEMPLATE = json.loads(gzip.decompress((FIXTURES / "family_templates.json.gz").read_bytes()))["inkling"]["chat_template"]
MODEL_TEMPLATE_SHA256 = "53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc"

END, STOP = "<|end_message|>", "<|content_model_end_sampling|>"
THINK = "<|message_model|><|content_thinking|>the capital of Italy" + END
TEXT = "<|message_model|><|content_text|>Rome" + END
CALL = '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Rome"}}' + END
SCHEMA = [{"name": "get_weather", "parameters": {"properties": {"city": {"type": "string"}}}}]


def read(reply, step, **kw):
    p = InklingOutputParser(**kw)
    return [e for i in range(0, len(reply), step) for e in p.feed(reply[i:i + step])] + p.finish()


def said(evs, kind):
    return "".join(e.text for e in evs if e.kind == kind)


class PreTokenizer(unittest.TestCase):
    def pieces(self, text):
        return ST.Tokenizer(["a"], [], pre="inkling2")._pieces(text)

    def test_the_pattern_is_gpt_4os(self):
        # digits in threes, a word with the capitals before it, a contraction with its word, ideographs with letters
        self.assertEqual(self.pieces("1234567 dollars"), ["123", "456", "7", " dollars"])
        self.assertEqual(self.pieces("HTTPServer and McDonald's"), ["HTTPServer", " and", " Mc", "Donald's"])
        self.assertEqual(self.pieces("they've  said.\n\n  x"), ["they've", " ", " said", ".\n\n", " ", " x"])
        self.assertEqual(self.pieces("你好ab"), ["你好ab"])
        self.assertEqual(self.pieces("a  \nb"), ["a", "  \n", "b"])

    def test_a_piece_that_is_a_token_is_that_token(self):
        # the model's tokenizer.json sets ignore_merges: "ab" is a token, though no merge makes it.  (In the real
        # vocabulary "kwargs" is such a token: Hugging Face gives [87550], llama.cpp [86226, 70, 82].)
        tok = ST.Tokenizer(["a", "b", "ab"], [], pre="inkling2")
        self.assertTrue(tok.ignore_merges)
        self.assertEqual(tok.encode("ab"), [2])


class Blocks(unittest.TestCase):
    def test_thinking_then_text_however_it_arrives(self):
        for step in (1, 7, 10_000):
            with self.subTest(step=step):
                evs = read(THINK + TEXT + STOP, step)
                self.assertEqual((said(evs, "reasoning"), said(evs, "content")), ("the capital of Italy", "Rome"))

    def test_a_call_of_a_declared_tool(self):
        for step in (1, 7, 10_000):
            with self.subTest(step=step):
                evs = read(THINK + CALL + STOP, step, tools=SCHEMA)
                calls = [e.call for e in evs if e.kind == "tool_call"]
                self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Rome"})])
                self.assertEqual(said(evs, "content"), "")

    def test_a_call_of_a_tool_nobody_declared_is_the_text_it_was_written_as(self):
        other = CALL.replace("get_weather", "rm")
        evs = read(other, 5, tools=SCHEMA)
        self.assertEqual([e.kind for e in evs if e.kind == "tool_call"], [])
        self.assertEqual(said(evs, "content"), other[:-len(END)])

    def test_a_call_that_is_not_json_is_text(self):
        broken = "<|message_model|>get_weather<|content_invoke_tool_json|>{city: Rome" + END
        evs = read(broken, 10_000, tools=SCHEMA)
        self.assertEqual(said(evs, "content"), broken[:-len(END)])

    def test_the_block_end_in_a_string_argument_does_not_end_the_call(self):
        # llama.cpp 29b52d398b reads this as the call; the tags' text can be in a string the model writes as text
        tools = [{"name": "python", "parameters": {"properties": {"code": {"type": "string"}}}}]
        call = '<|message_model|>python<|content_invoke_tool_json|>{"name":"python","args":{"code":"print(\'<|end_message|>\')"}}'
        for step in (1, 4, 10_000):
            with self.subTest(step=step):
                evs = read(call + END + TEXT + STOP, step, tools=tools)
                self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"],
                                 [{"code": "print('<|end_message|>')"}])
                self.assertEqual(said(evs, "content"), "Rome")
        # JSON that never closes: at the end, the block ends at its first tag and is the text it was written as
        broken = '<|message_model|>python<|content_invoke_tool_json|>{"name":"python","args":{"code":"x' + END
        evs = read(broken + TEXT, 3, tools=tools)
        self.assertFalse([e for e in evs if e.kind == "tool_call"])
        self.assertEqual(said(evs, "content"), broken[:-len(END)] + "Rome")

    def test_an_escaped_quote_does_not_end_the_string_that_holds_the_tags(self):
        # PR 211: \" inside the string, then both tags' text, then a second call right after the first
        tools = [{"name": "python", "parameters": {"properties": {"code": {"type": "string"}}}}]
        code = 'print("a \\"' + END + '<|message_model|>\\" b\\\\")'
        call = ('<|message_model|>python<|content_invoke_tool_json|>'
                + json.dumps({"name": "python", "args": {"code": code}}, ensure_ascii=False) + END)
        for step in (1, 5, 10_000):
            with self.subTest(step=step):
                evs = read(call + call + TEXT + STOP, step, tools=tools)
                self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"code": code}] * 2)
                self.assertEqual(said(evs, "content"), "Rome")

    def test_text_written_right_after_the_block_start_shows_no_tags(self):
        # #283: after the budget's wrap-up opened a block, the model wrote its answer with no <|content_text|>
        for step in (1, 3, 1000):
            with self.subTest(step=step):
                evs = read(THINK + "<|message_model|> there! Happy to help.\nMore." + END + STOP, step)
                self.assertEqual(said(evs, "content"), " there! Happy to help.\nMore.")
                self.assertEqual(said(evs, "content") + said(evs, "reasoning"),
                                 said(evs, "content").replace("<|", "") + said(evs, "reasoning"))
                self.assertEqual(said(read("<|message_model|>Rome" + END, step), "content"), "Rome")

    def test_a_reply_whose_text_block_the_prompt_opened(self):
        evs = read("Rome" + END + STOP, 3, opened=True)
        self.assertEqual((said(evs, "reasoning"), said(evs, "content")), ("", "Rome"))

    def test_a_block_cut_off_keeps_what_it_had(self):
        evs = read(THINK + "<|message_model|><|content_text|>Ro", 4)
        self.assertEqual((said(evs, "reasoning"), said(evs, "content")), ("the capital of Italy", "Ro"))

    def test_a_forced_call_opens_the_block_up_to_its_arguments(self):
        opening = inkling_forced_call(FUNC_START + "get_weather>")
        self.assertEqual(opening, '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":')
        self.assertIsNone(inkling_forced_call(None))
        evs = read(opening + '{"city":"Rome"}}' + END, 6, tools=SCHEMA)
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                         [("get_weather", {"city": "Rome"})])


class InklingTokens(ByteTokenizer):
    SPECIALS = ["<|message_system|>", "<|message_user|>", "<|message_tool|>", "<|message_model|>", "<|content_text|>",
                "<|content_thinking|>", "<|content_invoke_tool_json|>", END, STOP]
    pre = "inkling2"
    special_ids = {"tokenizer.ggml.eos_token_id": 264}


# Inkling 2's template, cut to what the server depends on: its one switch, the effort, in a system message
TEMPLATE = ('{{- "<|message_system|>effort=" ~ reasoning_effort ~ "<|end_message|>" -}}'
            '{%- for m in messages -%}{{- "<|message_user|>" + m.content + "<|end_message|>" -}}{%- endfor -%}')


class ServiceForInkling(unittest.TestCase):
    def setUp(self):
        self.tok = InklingTokens()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chat_template.jinja"
            path.write_text(TEMPLATE, encoding="utf-8")
            self.svc = Service(MockEngine(self.tok, "hi"), self.tok, ChatTemplate(path))
        self.chat = [{"role": "user", "content": "capital of Italy?"}]

    def test_its_blocks_are_read_by_its_own_parser(self):
        self.assertEqual(self.svc.call_format, "inkling")
        self.assertIs(self.svc.parser_cls, InklingOutputParser)
        self.assertEqual(self.svc.text_open_ids, [259, 260])    # <|message_model|><|content_text|>
        self.assertIn(264, self.svc.stop_ids)

    def test_the_effort_the_template_is_given(self):
        for kwargs, effort in (({}, "high"), ({"reasoning_effort": "low"}, "low"), ({"reasoning_effort": 0.5}, "0.5"),
                               ({"reasoning_effort": "ludicrous"}, "high"), ({"enable_thinking": False}, "none")):
            with self.subTest(**kwargs):
                text = self.svc.render_prompt(self.chat, None, kwargs)
                self.assertTrue(text.startswith(f"<|message_system|>effort={effort}<|end_message|>"), text)
                self.assertEqual(text.endswith(INK_TEXT_OPEN), kwargs.get("enable_thinking") is False)

    def test_a_shared_prefix_is_the_first_messages_whatever_the_effort(self):
        # strata_prefix renders the first messages with the effort render_prompt writes: the common start is the
        # whole first message, not the few tokens before the effort line differs
        msgs = [{"role": "user", "content": "DOC " * 40}, {"role": "user", "content": "Which word repeats?"}]
        for kwargs in ({}, {"reasoning_effort": "low"}, {"enable_thinking": False}, {"reasoning_effort": "ludicrous"}):
            with self.subTest(**kwargs):
                req = {"strata_prefix": {"messages": 1}}
                ids, _, _ = self.svc.prepare(msgs, None, dict(kwargs), 16, req=req)
                head = self.tok.decode(ids[:req["strata_prefix"].get("tokens", 0)])
                self.assertTrue(head.endswith("DOC <|end_message|>"), head[-60:])

    def test_a_shared_prefix_with_a_numeric_effort_or_thinking_off_over_one(self):
        # PR 213: a number is an effort of its own, and thinking off wins over a named effort, as in render_prompt
        msgs = [{"role": "user", "content": "DOC " * 40}, {"role": "user", "content": "Which word repeats?"}]
        for kwargs, effort in (({"reasoning_effort": 0.5}, "0.5"),
                               ({"reasoning_effort": "max", "enable_thinking": False}, "none")):
            with self.subTest(**kwargs):
                req = {"strata_prefix": {"messages": 1}}
                ids, _, _ = self.svc.prepare(msgs, None, dict(kwargs), 16, req=req)
                head = self.tok.decode(ids[:req["strata_prefix"].get("tokens", 0)])
                self.assertEqual(head, f"<|message_system|>effort={effort}<|end_message|>"
                                       f"<|message_user|>{'DOC ' * 40}<|end_message|>")
                self.assertTrue(self.svc.render_prompt(msgs, None, dict(kwargs)).startswith(head))

    def test_a_reply_goes_on_in_the_text_block_a_prompt_opened(self):
        for kwargs, reply, reasoning in (({}, THINK + TEXT + STOP, "the capital of Italy"),
                                         ({"enable_thinking": False}, "Rome" + END + STOP, "")):
            with self.subTest(**kwargs):
                ids = self.svc.encode_prompt(self.chat, None, kwargs)
                self.svc.engine = MockEngine(self.tok, reply)
                evs = [e for kind, e in self.svc.run(ids, not kwargs, None, 64, {}, threading.Event())
                       if kind == "event"]
                self.assertEqual((said(evs, "reasoning"), said(evs, "content")), (reasoning, "Rome"))

    def test_a_named_tool_choice_is_opened_once_the_thinking_is_closed(self):
        # the model thinks; the server writes the call's opening right after the closed thinking block, in place of
        # the text block the model would open, and the model goes on with the arguments
        tools = [{"name": "get_weather", "parameters": {"properties": {"city": {"type": "string"}}}}]
        force = self.svc.force_text(FUNC_START + "get_weather>\n")
        ids = self.svc.encode_prompt(self.chat, tools, {})
        self.svc.engine = MockEngine(self.tok, [THINK + "<|message_model|><|content_text|>No tool needed.",
                                                '{"city":"Rome"}}' + END + STOP])
        evs = [e for kind, e in self.svc.run(ids, True, tools, 256, {}, threading.Event(), force) if kind == "event"]
        forced = self.tok.encode(force, parse_special=True)
        self.assertEqual(self.svc.engine.last_prompt[-len(forced):], forced)
        self.assertEqual(self.tok.decode(self.svc.engine.last_prompt[len(ids):]), THINK + force)
        self.assertEqual(said(evs, "reasoning"), "the capital of Italy")
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                         [("get_weather", {"city": "Rome"})])
        self.assertEqual(said(evs, "content"), "")

    def test_the_parser_says_where_a_forced_call_can_go(self):
        p = InklingOutputParser(tools=SCHEMA)
        p.feed("<|message_model|><|content_thinking|>x")
        self.assertEqual(p.state, "reasoning")
        p.feed(END)
        self.assertEqual((p.state, p.clean), ("content", True))     # between blocks
        p.feed("<|message_model|><|content_text|>y")
        self.assertEqual(p.state, "text")                            # not inside a text block

    def test_the_thinking_budget_closes_the_thinking_block(self):
        # #273: the budget used to be dropped for Inkling 2, so a model that thought in circles wrote until max_tokens
        # and the reply had no answer.  At the budget the server ends the thinking block as the model would and opens
        # the next one; the model writes the text block.
        ids = self.svc.encode_prompt(self.chat, None, {})
        self.svc.engine = MockEngine(self.tok, ["<|message_model|><|content_thinking|>" + "round and round " * 20,
                                                "<|content_text|>Rome" + END + STOP])
        evs = [e for kind, e in self.svc.run(ids, True, None, 512, {"reasoning_budget_tokens": 40}, threading.Event())
               if kind == "event"]
        written = self.tok.decode(self.svc.engine.last_prompt[len(ids):])
        self.assertTrue(written.endswith(END + "<|message_model|>"), written[-80:])
        self.assertNotIn("</think>", written)
        self.assertEqual(said(evs, "content"), "Rome")
        self.assertNotIn("</think>", said(evs, "reasoning"))

    def test_reasoning_tokens_are_the_thinking_block_s(self):
        # PR 212: the text block's tokens are not thinking tokens (usage's reasoning_tokens)
        ids = self.svc.encode_prompt(self.chat, None, {})
        self.svc.engine = MockEngine(self.tok, THINK + TEXT + STOP)
        done = [e for kind, e in self.svc.run(ids, True, None, 64, {}, threading.Event()) if kind == "done"][0]
        self.assertEqual(done["reasoning_tokens"], len(self.tok.encode("the capital of Italy" + END, parse_special=True)))


def model_template() -> ChatTemplate:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "chat_template.jinja"
        path.write_text(MODEL_TEMPLATE, encoding="utf-8")
        return ChatTemplate(path)


def subset_tokenizer():
    """The vocabulary's tokens and merges that the golden strings use (serve/fixtures/inkling2_vocab_subset.json), at
    their own ids; the other ids hold placeholders."""
    sub = json.loads((FIXTURES / "inkling2_vocab_subset.json").read_text(encoding="utf-8"))
    tokens = [f"<placeholder {i}>" for i in range(sub["n_tokens"])]
    types = [1] * sub["n_tokens"]
    for i, t in sub["tokens"].items():
        tokens[int(i)] = t
    for i in sub["control"]:
        types[i] = 3
    return ST.Tokenizer(tokens, sub["merges"], types, pre=sub["pre"], special_ids=sub["special_ids"])


class GoldenIds(unittest.TestCase):
    """Hugging Face's ids for the golden strings, with and without parse_special (no download needed)."""

    def test_the_golden_strings(self):
        tok = subset_tokenizer()
        self.assertGreater(len(GOLDEN["tokens"]), 40)
        for case in GOLDEN["tokens"]:
            with self.subTest(case["name"]):
                ids = tok.encode(case["text"], parse_special=case["parse_special"])
                self.assertEqual(ids, case["ids"])
                self.assertEqual(tok.decode(ids), case["text"])

    def test_a_prompt_is_cut_at_the_control_tokens(self):
        tok = subset_tokenizer()
        ids = tok.encode("<|message_user|><|content_text|>Hi there<|end_message|><|message_model|>", parse_special=True)
        self.assertEqual((ids[:2], ids[-2:]), ([200000, 200004], [200010, 200001]))
        self.assertNotIn(200010, tok.encode("<|end_message|>"))     # text unless parse_special

    def test_where_llama_cpp_differs(self):
        # "kwargs" is one token that no chain of merges reaches: Hugging Face (ignore_merges) gives it whole,
        # llama.cpp 29b52d398b as three
        case = next(c for c in GOLDEN["tokens"] if "llama_cpp_ids" in c)
        self.assertEqual(case["text"], "kwargs **kwargs")
        self.assertEqual(subset_tokenizer().encode(case["text"]), [87550, 197033, 87550])
        self.assertEqual(case["llama_cpp_ids"], [86226, 70, 82, 197033, 86226, 70, 82])


def real_tokenizer():
    path = os.environ.get("STRATA_INKLING_TOKENIZER")
    return ST.Tokenizer.from_dir(path) if path and (Path(path) / "vocab.json").is_file() else None


@unittest.skipIf(real_tokenizer() is None, "no Inkling tokenizer here ($STRATA_INKLING_TOKENIZER)")
class RealVocabulary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = real_tokenizer()

    def test_the_vocabulary_and_its_specials(self):
        self.assertEqual(len(self.tok.tokens), 200058)
        self.assertEqual(self.tok.pre, "inkling2")
        self.assertEqual(self.tok.special_ids["tokenizer.ggml.eos_token_id"], 200006)
        self.assertNotIn("tokenizer.ggml.bos_token_id", self.tok.special_ids)
        for text, i in (("<|message_model|>", 200001), ("<|content_model_end_sampling|>", 200006),
                        ("<|end_message|>", 200010), ("<|content_invoke_tool_json|>", 200049)):
            self.assertEqual(self.tok.encode(text, parse_special=True), [i])
            self.assertNotEqual(self.tok.encode(text), [i])      # a control token: text unless parse_special

    def test_round_trips(self):
        for text in ("", "Hello, world!", "  leading and trailing  ", "def f(x):\n\treturn x\n", "你好，世界",
                     "مرحبا", "\U0001f600\U0001f680\U0001f1fa\U0001f1f8", "é́", "\x00\x01\x7f",
                     " nbsp", "1234567890", "\r\n\r\n", "x" * 3000, "<|message_user|> as text"):
            with self.subTest(text=text[:20]):
                self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_the_golden_ids(self):
        for case in GOLDEN["tokens"]:
            with self.subTest(case["name"]):
                self.assertEqual(self.tok.encode(case["text"], parse_special=case["parse_special"]), case["ids"])


class ModelTemplate(unittest.TestCase):
    """The model's own template: what it renders for system prompts, tools, calls and their answers, and llama.cpp's
    prompts for the same OpenAI requests."""
    TOOLS = [{"name": "get_weather", "description": "Get the weather",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                             "required": ["city"]}}]

    @classmethod
    def setUpClass(cls):
        cls.t = model_template()

    def test_it_is_the_models(self):
        self.assertEqual(hashlib.sha256(MODEL_TEMPLATE.encode("utf-8")).hexdigest(), MODEL_TEMPLATE_SHA256)

    def test_the_effort_comes_after_the_system_prompt_and_reads_names_and_numbers(self):
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
        for effort, shown in (("none", "0"), ("minimal", "0.1"), ("low", "0.2"), ("medium", "0.7"), ("high", "0.9"),
                              ("xhigh", "0.99"), ("max", "0.99"), (0.5, "0.5")):
            with self.subTest(effort=effort):
                self.assertEqual(self.t.render(msgs, reasoning_effort=effort),
                                 "<|message_system|><|content_text|>Be brief.<|end_message|>"
                                 f"<|message_system|><|content_text|>Thinking effort level: {shown}<|end_message|>"
                                 "<|message_user|><|content_text|>Hi<|end_message|>")

    def test_tools_calls_and_answers(self):
        msgs = [{"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": "", "reasoning_content": "I need the tool.",
                 "tool_calls": [{"id": "c1", "function": {"name": "get_weather",
                                                          "arguments": {"days": 2, "city": "Paris"}}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "18 C"},
                {"role": "assistant", "content": "It is 18 C."}]
        self.assertEqual(self.t.render(msgs, tools=self.TOOLS),
                         '<|message_system|>tool_declare<|content_xml|>[{"description":"Get the weather",'
                         '"name":"get_weather","parameters":{"properties":{"city":{"type":"string"},'
                         '"days":{"type":"integer"}},"required":["city"],"type":"object"},"type":"function"}]'
                         "<|end_message|>"
                         "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                         "<|message_user|><|content_text|>Weather in Paris?<|end_message|>"
                         "<|message_model|><|content_thinking|>I need the tool.<|end_message|>"
                         '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather",'
                         '"args":{"city":"Paris","days":2}}<|end_message|><|content_model_end_sampling|>'
                         "<|message_tool|>get_weather<|content_text|>18 C<|end_message|>"
                         "<|message_model|><|content_text|>It is 18 C.<|end_message|><|content_model_end_sampling|>")

    def test_a_tool_in_openai_shape_renders_the_same(self):
        flat = self.t.render([{"role": "user", "content": "x"}], tools=self.TOOLS)
        wrapped = self.t.render([{"role": "user", "content": "x"}],
                                tools=[{"type": "function", "function": self.TOOLS[0]}])
        self.assertEqual(flat, wrapped)

    def test_llama_cpp_renders_the_same(self):
        self.assertGreater(len(GOLDEN["templates"]), 10)
        for case in GOLDEN["templates"]:
            with self.subTest(case["name"]):
                messages, tools, _ = openai_to_messages(case["request"])
                self.assertEqual(self.t.render(messages, tools=tools, **case["kwargs"]), case["prompt"])


H, CTHINK, CTEXT, CJSON = ("<|message_model|>", "<|content_thinking|>", "<|content_text|>",
                           "<|content_invoke_tool_json|>")


def events(text, step, tools=None, stream_tools=True, thinking=True):
    p = InklingOutputParser(thinking=thinking, tools=tools, stream_tools=stream_tools)
    return [e for i in range(0, len(text), step) for e in p.feed(text[i:i + step])] + p.finish()


class Replies(unittest.TestCase):
    TOOLS = [{"name": "get_weather", "parameters": {"properties": {"city": {"type": "string"}}}},
             {"name": "now", "parameters": {"properties": {}}}]
    STEPS = (1, 3, 7, 10_000)

    def test_llama_cpp_reads_the_same(self):
        # llama.cpp's parse of the replies in its own test-chat.cpp cases for Inkling 2, and a few more
        self.assertGreater(len(GOLDEN["replies"]), 10)
        for case in GOLDEN["replies"]:
            for step in (1, 5, 10_000):
                with self.subTest(case["name"], step=step):
                    evs = events(case["text"], step, case["tools"] or None)
                    self.assertEqual(said(evs, "content"), case["content"])
                    self.assertEqual(said(evs, "reasoning"), case["reasoning"])
                    self.assertEqual([{"name": e.call.name, "arguments": e.call.arguments}
                                      for e in evs if e.kind == "tool_call"], case["calls"])

    def test_blocks_one_after_another(self):
        text = f"{H}{CTEXT}One.{END}{H}{CTHINK}more{END}{H}{CTEXT} Two.{END}"
        for step in self.STEPS:
            with self.subTest(step=step):
                evs = events(text, step)
                self.assertEqual((said(evs, "content"), said(evs, "reasoning")), ("One. Two.", "more"))

    def test_thinking_off_still_reads_a_thinking_block_as_reasoning(self):
        evs = events(f"{H}{CTHINK}brief{END}{H}{CTEXT}ok{END}", 2, thinking=False)
        self.assertEqual((said(evs, "reasoning"), said(evs, "content")), ("brief", "ok"))

    def test_two_calls_however_they_arrive(self):
        text = (f"{H}{CTHINK}Need weather.{END}{H}get_weather{CJSON}"
                '{"name":"get_weather","args":{"city":"Paris <|x|>"}}' f"{END}{H}now{CJSON}" '{"name":"now","args":{}}'
                f"{END}")
        for stream_tools in (False, True):
            for step in self.STEPS:
                with self.subTest(stream_tools=stream_tools, step=step):
                    evs = events(text, step, self.TOOLS, stream_tools)
                    self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                                     [("get_weather", {"city": "Paris <|x|>"}), ("now", {})])
                    self.assertEqual((said(evs, "reasoning"), said(evs, "content")), ("Need weather.", ""))

    def test_a_streamed_call_is_announced_and_keeps_its_id(self):
        text = f"{H}get_weather{CJSON}" '{"name":"get_weather","args":{"city":"Oslo"}}' f"{END}"
        evs = [e for e in events(text, 1, self.TOOLS) if e.kind.startswith("tool_")]
        self.assertEqual([e.kind for e in evs], ["tool_start", "tool_args", "tool_call"])
        self.assertEqual(evs[0].call.id, evs[2].call.id)
        self.assertEqual(json.loads(evs[1].text), {"city": "Oslo"})

    def test_a_call_whose_end_tag_did_not_come_before_the_next_block(self):
        for step in (1, 4, 1000):
            with self.subTest(step=step):
                evs = events(f"{H}now{CJSON}" '{"name":"now","args":{}}' f"{H}{CTEXT}Done.{END}", step, self.TOOLS)
                self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                                 [("now", {})])
                self.assertEqual(said(evs, "content"), "Done.")

    def test_a_call_cut_off_in_its_json_is_no_call(self):
        evs = events(f"{H}get_weather{CJSON}" '{"name":"get_weather","args":{"ci', 5, self.TOOLS)
        self.assertFalse([e for e in evs if e.kind == "tool_call"])

    def test_stray_markers_are_not_shown(self):
        evs = events(f"{H}{CTEXT}Hi{END}{STOP}", 1)
        self.assertEqual(said(evs, "content"), "Hi")

    def test_a_block_without_its_content_token_is_text(self):
        evs = events(f"{H}Plain answer.\nSecond line{END}", 3)
        self.assertIn("Plain answer.\nSecond line", said(evs, "content"))


class FullInklingTokens(ByteTokenizer):
    # the model's control tokens the template writes, in the vocabulary's order (ids 256...)
    SPECIALS = ["<|endoftext|>", "<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>",
                "<|content_text|>", "<|content_image|>", "<|content_model_end_sampling|>", "<|content_thinking|>",
                "<|end_message|>", "<|content_xml|>", "<|content_invoke_tool_json|>"]
    pre = "inkling2"
    special_ids = {"tokenizer.ggml.eos_token_id": 263, "tokenizer.ggml.eot_token_id": 263,
                   "tokenizer.ggml.padding_token_id": 256}


class ServiceWithTheModelTemplate(unittest.TestCase):
    TOOLS = [{"name": "get_weather", "parameters": {"properties": {"city": {"type": "string"}}}}]

    def setUp(self):
        self.tok = FullInklingTokens()
        self.svc = Service(MockEngine(self.tok, "hi"), self.tok, model_template())

    def prompt(self, req):
        return self.svc.render_prompt(*openai_to_messages(req))

    def test_stops_and_turns(self):
        self.assertIn(263, self.svc.stop_ids)                         # <|content_model_end_sampling|>
        self.assertEqual(self.svc.turn_ids, frozenset({257, 258, 259, 260}))

    def test_the_effort_a_client_asks_for(self):
        user = [{"role": "user", "content": "Hi"}]
        for req, shown in (({}, "0.9"), ({"reasoning_effort": "low"}, "0.2"), ({"reasoning_effort": "medium"}, "0.7"),
                           ({"reasoning_effort": "xhigh"}, "0.99")):
            with self.subTest(req=req):
                self.assertEqual(self.prompt({"messages": user, **req}),
                                 f"<|message_system|><|content_text|>Thinking effort level: {shown}<|end_message|>"
                                 "<|message_user|><|content_text|>Hi<|end_message|>")
        for req in ({"reasoning_effort": "none"}, {"chat_template_kwargs": {"enable_thinking": False}}):
            with self.subTest(req=req):
                off = self.prompt({"messages": user, **req})
                self.assertEqual(off, "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
                                      "<|message_user|><|content_text|>Hi<|end_message|>" + INK_TEXT_OPEN)
        messages, tools, kwargs = anthropic_to_messages({"messages": user, "thinking": {"type": "disabled"}})
        self.assertIn("Thinking effort level: 0<", self.svc.render_prompt(messages, tools, kwargs))

    def test_a_system_prompt_and_tools_come_before_the_effort(self):
        text = self.prompt({"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
                            "tools": [{"type": "function", "function": self.TOOLS[0]}]})
        self.assertTrue(text.startswith("<|message_system|>tool_declare<|content_xml|>[{"), text)
        self.assertLess(text.index("Be brief."), text.index("Thinking effort level: 0.9"))

    def test_a_control_token_in_a_message_is_text(self):
        ids = self.svc.encode_prompt(*openai_to_messages({"messages": [{"role": "user", "content": "<|end_message|>"}]}))
        self.assertEqual(ids.count(265), 2)                           # the template's own two, not the message's
        self.assertIn(list(b"<|end_message|>"), [ids[i:i + 15] for i in range(len(ids))])

    def run_reply(self, reply, tools=None, force=None, thinking=True):
        kwargs = {} if thinking else {"enable_thinking": False}
        ids, _, _ = self.svc.prepare([{"role": "user", "content": "q"}], tools, kwargs, 256, force=force)
        self.svc.engine = MockEngine(self.tok, reply)
        return [e for kind, e in self.svc.run(ids, thinking, tools, 256, {}, threading.Event(), force)
                if kind == "event"]

    def test_a_tool_call_through_the_service(self):
        evs = self.run_reply(f"{H}get_weather{CJSON}" '{"name":"get_weather","args":{"city":"Rome"}}' f"{END}{STOP}",
                             self.TOOLS)
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                         [("get_weather", {"city": "Rome"})])

    def test_a_forced_call_without_thinking_replaces_the_text_block(self):
        force = self.svc.force_text(forced_call({"type": "function", "function": {"name": "get_weather"}}, self.TOOLS))
        ids, _, _ = self.svc.prepare([{"role": "user", "content": "q"}], self.TOOLS, {"enable_thinking": False}, 64,
                                     force=force)
        text = self.tok.decode(ids)
        self.assertTrue(text.endswith(f"<|message_user|><|content_text|>q{END}" + force), text[-120:])


class Pictures(unittest.TestCase):
    """A picture in a user or tool turn: the template writes <|content_image|><|unused_200054|> in a block of its own,
    and the server puts one <|unused_200054|> a row of the picture there, as llama.cpp 29b52d398b's mtmd feeds it."""

    class FakeVision:
        def __init__(self, d, n=(3, 2)):
            self.n, self.seen = list(n), {}
            self.dir = Path(d)
            self.rows = self.dir / "img.sve"
            self.rows.write_bytes(b"rows")

        def encode(self, source):
            if source not in self.seen:
                self.seen[source] = self.n.pop(0)
            return self.rows, self.seen[source]

    def service(self, d, n=(3, 2)):
        class Asked(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                self.asked = (StrataEngine.sampling_keys(sampling), embeddings)
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = subset_tokenizer()
        return Service(Asked(tok, "a cat"), tok, model_template(), vision=self.FakeVision(d, n))

    def test_the_rows_go_where_the_template_put_the_picture(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self.service(d)
            msgs = [{"role": "user", "content": [{"type": "image", "source": "a.png"},
                                                 {"type": "text", "text": "Hi there"},
                                                 {"type": "image", "source": "b.png"}]}]
            ids, _, _ = svc.prepare(msgs, None, {})
            at = ids.index(200005)
            self.assertEqual(ids[at - 1:at + 7], [200000, 200005, 200054, 200054, 200054, 200010, 200000, 200004])
            at = ids.index(200005, at + 1)
            self.assertEqual(ids[at - 1:at + 4], [200000, 200005, 200054, 200054, 200010])
            self.assertEqual((ids.count(200054), svc.embeddings.pad), (5, 200054))
            rows = svc.embeddings.path
            list(svc.run(ids, False, None, 8, {}, threading.Event()))
            self.assertEqual(svc.engine.asked, (" image_pad=200054", rows))

    def test_two_pictures_back_to_back(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self.service(d, (2, 3))
            msgs = [{"role": "user", "content": [{"type": "image", "source": "a.png"},
                                                 {"type": "image", "source": "b.png"}]}]
            ids, _, _ = svc.prepare(msgs, None, {})
            at = ids.index(200005)
            self.assertEqual(ids[at - 1:at + 10], [200000, 200005, 200054, 200054, 200010,
                                                   200000, 200005, 200054, 200054, 200054, 200010])

    def test_a_picture_in_a_tool_message(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self.service(d, (3,))
            msgs = [{"role": "user", "content": "Hi there"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "shot", "arguments": {}}}]},
                    {"role": "tool", "tool_call_id": "c1",
                     "content": [{"type": "image", "source": "a.png"}, {"type": "text", "text": "Hi there"}]}]
            ids, _, _ = svc.prepare(msgs, None, {})
            tool = svc.tok.encode("<|message_tool|>shot", parse_special=True)
            at = next(i for i in range(len(ids)) if ids[i:i + len(tool)] == tool) + len(tool)
            self.assertEqual(ids[at:at + 6], [200005, 200054, 200054, 200054, 200010] + tool[:1])
            self.assertEqual((ids.count(200054), svc.embeddings.pad), (3, 200054))

    def test_the_marks_in_a_message_are_text(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self.service(d, (2,))
            msgs = [{"role": "user", "content": [{"type": "text", "text": "<|content_image|><|unused_200054|>"},
                                                 {"type": "image", "source": "a.png"}]}]
            ids, _, _ = svc.prepare(msgs, None, {})
            self.assertEqual((ids.count(200005), ids.count(200054)), (1, 2))
            self.assertIn(svc.tok.encode("<|unused_200054|>"), [ids[i:i + 15] for i in range(len(ids))])

    def test_a_picture_from_the_model_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self.service(d, (2,))
            msgs = [{"role": "user", "content": "Hi there"},
                    {"role": "assistant", "content": [{"type": "image", "source": "a.png"}]},
                    {"role": "user", "content": "Hi there"}]
            with self.assertRaises(Exception) as e:
                svc.prepare(msgs, None, {})
            self.assertIn("only supported for user and tool messages", str(e.exception))


if __name__ == "__main__":
    unittest.main()
