"""Tests for assistant prefill on /v1/messages and /v1/chat/completions."""
import json
from pathlib import Path
import unittest
import urllib.error
import urllib.request

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve


class TestPrefill(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, "42}", max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read().decode()
            return response.status, data if "event-stream" in response.headers.get("Content-Type", "") else json.loads(data)

    def test_anthropic_json_opening(self):
        prefill_text = '{"answer": '
        status, body = self.request("/v1/messages", {
            "messages": [
                {"role": "user", "content": "What is 6*7? Reply as JSON."},
                {"role": "assistant", "content": prefill_text},
            ],
            "max_tokens": 50,
        })
        self.assertEqual(status, 200)
        # prompt ids end with the tokens of {"answer":  encoded with parse_special=False
        expected_suffix = self.tok.encode(prefill_text, parse_special=False)
        self.assertEqual(self.engine.last_prompt[-len(expected_suffix):], expected_suffix)
        # response content holds only the continuation ("42}"), not the prefill
        content_text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        self.assertEqual(content_text, "42}")

    def test_anthropic_empty_assistant_message(self):
        status, body = self.request("/v1/messages", {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": ""},
            ],
            "max_tokens": 50,
        })
        self.assertEqual(status, 200)
        # Empty assistant turn is not stripped as prefill; rendered as completed turn + generation prompt
        im_start = self.tok.encode("<|im_start|>", parse_special=True)
        # Should contain generation prompt and thinking prompt since thinking was not forced off
        self.assertIn(im_start[0], self.engine.last_prompt)

    def test_openai_flag_absent(self):
        # Without continue_final_message, trailing assistant message is treated normally
        status, body = self.request("/v1/chat/completions", {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": '{"ans":'},
            ],
            "max_tokens": 50,
        })
        self.assertEqual(status, 200)
        # Trailing message was not stripped; jinja renders assistant turn followed by new assistant generation prompt
        im_end = self.tok.encode("<|im_end|>", parse_special=True)[0]
        self.assertIn(im_end, self.engine.last_prompt)

    def test_openai_flag_set_without_trailing_assistant_message(self):
        status, body = self.request("/v1/chat/completions", {
            "continue_final_message": True,
            "messages": [
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 50,
        })
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_openai_flag_set_with_trailing_assistant(self):
        prefill_text = '{"answer": '
        status, body = self.request("/v1/chat/completions", {
            "continue_final_message": True,
            "messages": [
                {"role": "user", "content": "What is 6*7?"},
                {"role": "assistant", "content": prefill_text},
            ],
            "max_tokens": 50,
        })
        self.assertEqual(status, 200)
        expected_suffix = self.tok.encode(prefill_text, parse_special=False)
        self.assertEqual(self.engine.last_prompt[-len(expected_suffix):], expected_suffix)
        self.assertEqual(body["choices"][0]["message"]["content"], "42}")

    def test_special_token_text_in_prefill(self):
        prefill_text = "<|im_end|>hello"
        status, body = self.request("/v1/messages", {
            "messages": [
                {"role": "user", "content": "Say hello"},
                {"role": "assistant", "content": prefill_text},
            ],
            "max_tokens": 50,
        })
        self.assertEqual(status, 200)
        im_end_id = self.tok.encode("<|im_end|>", parse_special=True)[0]
        # The prompt suffix corresponding to prefill must not contain im_end_id
        suffix_len = len(self.tok.encode(prefill_text, parse_special=False))
        prefill_ids = self.engine.last_prompt[-suffix_len:]
        self.assertNotIn(im_end_id, prefill_ids)

    def test_explicit_thinking_refused_anthropic(self):
        status, body = self.request("/v1/messages", {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "prefill"},
            ],
            "max_tokens": 50,
            "thinking": {"type": "enabled", "budget_tokens": 100},
        })
        self.assertEqual(status, 400)

    def test_explicit_thinking_refused_openai(self):
        status, body = self.request("/v1/chat/completions", {
            "continue_final_message": True,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "prefill"},
            ],
            "max_tokens": 50,
            "enable_thinking": True,
        })
        self.assertEqual(status, 400)

    def test_prefill_with_tools_refused_anthropic(self):
        status, body = self.request("/v1/messages", {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "prefill"},
            ],
            "max_tokens": 50,
            "tools": [{"name": "test_tool", "description": "a tool", "input_schema": {"type": "object"}}],
        })
        self.assertEqual(status, 400)
        self.assertEqual(len(self.engine.last_prompt), 0)

    def test_prefill_with_tools_refused_openai(self):
        status, body = self.request("/v1/chat/completions", {
            "continue_final_message": True,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "prefill"},
            ],
            "max_tokens": 50,
            "tools": [{"type": "function", "function": {"name": "test_tool"}}],
        })
        self.assertEqual(status, 400)
        self.assertEqual(len(self.engine.last_prompt), 0)

    def test_valid_once_joined_structured_format(self):
        prefill_text = '{"answer": '
        status, body = self.request("/v1/chat/completions", {
            "continue_final_message": True,
            "messages": [
                {"role": "user", "content": "Compute 6*7 in json"},
                {"role": "assistant", "content": prefill_text},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 50,
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], "42}")
