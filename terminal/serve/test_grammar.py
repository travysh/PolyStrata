import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import unittest.mock
import urllib.error
import urllib.request

from serve import grammar as grammar_api
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve

GBNF = 'root ::= "{\\"parts\\":[" [a-z]+ "]}"\n'


class Recording(MockEngine):
    """A MockEngine that reports grammar support and keeps what the route sent it."""

    def __init__(self, *a, info=None, **kw):
        super().__init__(*a, **kw)
        self.info = {"grammar": 1} if info is None else info
        self.sampling = None

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sampling = sampling
        return super().generate(ids, max_new, sampling, cancel, embeddings)


class Validate(unittest.TestCase):
    ok = SimpleNamespace(info={"grammar": 1})

    def test_none_without_the_field(self):
        self.assertIsNone(grammar_api.validate_request({}, SimpleNamespace(info={}), None))
        self.assertIsNone(grammar_api.validate_request({"grammar": None}, SimpleNamespace(info={}), None))

    def test_accepted(self):
        g = grammar_api.validate_request({"grammar": GBNF, "response_format": {"type": "text"},
                                          "reasoning_effort": "none", "tool_choice": "none"}, self.ok, None)
        self.assertIsInstance(g, grammar_api.Checked)
        self.assertEqual(g, GBNF)

    def test_bad_values(self):
        for bad in [1, True, [], {}, "", "  \n"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                grammar_api.validate_request({"grammar": bad}, self.ok, None)
        grammar_api.validate_request({"grammar": "a" * grammar_api.MAX_BYTES}, self.ok, None)
        with self.assertRaisesRegex(ValueError, "64 KiB"):
            grammar_api.validate_request({"grammar": "a" * (grammar_api.MAX_BYTES + 1)}, self.ok, None)

    def test_engine_gate(self):
        for engine in [SimpleNamespace(), SimpleNamespace(info={}), SimpleNamespace(info={"grammar": 0})]:
            with self.assertRaisesRegex(ValueError, "does not support grammars"):
                grammar_api.validate_request({"grammar": GBNF}, engine, None)
        with self.assertRaisesRegex(ValueError, "--batch"):
            grammar_api.validate_request({"grammar": GBNF}, SimpleNamespace(info={"grammar": 1}, batch=2), None)

    def test_combinations(self):
        tool = {"type": "function", "function": {"name": "f", "parameters": {}}}
        cases = {
            "tools": {"tools": [tool]},
            "forced tool_choice": {"tool_choice": {"type": "function", "function": {"name": "f"}}},
            "required tool_choice": {"tool_choice": "required"},
            "mcp": {"strata_mcp": True},
            "prefill": {"continue_final_message": True},
            "enable_thinking": {"enable_thinking": True},
            "template kwargs": {"chat_template_kwargs": {"enable_thinking": True}},
            "reasoning_effort": {"reasoning_effort": "high"},
            "reasoning": {"reasoning": {"effort": "medium"}},
            "json_object": {"response_format": {"type": "json_object"}},
            "json_schema": {"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}},
        }
        for name, extra in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                grammar_api.validate_request({"grammar": GBNF, **extra}, self.ok, None)

    def test_transport(self):
        g = grammar_api.Checked(GBNF)
        key = grammar_api.engine_key(g)
        self.assertTrue(key.startswith(" grammar="))
        self.assertEqual(base64.b64decode(key[len(" grammar="):]).decode(), GBNF)
        self.assertNotIn(" ", key[1:])                 # one GEN token, whatever the grammar's spaces and newlines
        self.assertEqual(StrataEngine.sampling_keys({"grammar": g}), key)
        for plain in [GBNF, None, ""]:                  # not validated by the route: ignored
            self.assertEqual(grammar_api.engine_key(plain), "")
            self.assertEqual(StrataEngine.sampling_keys({"grammar": plain}), StrataEngine.sampling_keys({}))
        self.assertEqual(grammar_api.engine_key(grammar_api.Checked("root ::= \"é\"")),
                         " grammar=" + base64.b64encode('root ::= "é"'.encode()).decode())


FIXTURES = Path(__file__).parent / 'fixtures'


class SchemaGrammar(unittest.TestCase):
    """from_schema(): llama.cpp's converter, on the add-on generator's fill schemas."""

    def grammars(self):
        out = {'json_object': grammar_api.from_schema({'type': 'object'})}
        for name in ('items_weapons', 'armor_sets'):
            out[name] = grammar_api.from_schema(json.loads((FIXTURES / f'fill_schema_{name}.json').read_text()))
        return out

    def test_converts(self):
        for name, g in self.grammars().items():
            with self.subTest(name):
                self.assertIsInstance(g, grammar_api.Checked)
                self.assertIn('\nroot ::= ', '\n' + g)
                self.assertIn('space ::= ', g)

    def test_property_order_and_required(self):
        g = grammar_api.from_schema({'type': 'object', 'properties': {'zeta': {'type': 'integer'},
                                                                     'alpha': {'type': 'string'}},
                                     'required': ['zeta', 'alpha'], 'additionalProperties': False})
        root = [line for line in g.split('\n') if line.startswith('root ::= ')][0]
        self.assertLess(root.index('zeta-kv'), root.index('alpha-kv'))   # the schema's order, not sorted

    def test_unconvertible_is_none(self):
        self.assertIsNone(grammar_api.from_schema({'$ref': 'ftp://nowhere/x.json'}))
        self.assertIsNone(grammar_api.from_schema({'type': 'object', 'properties': {'a': {'$ref': 'https://x/y'}}}))

    def test_the_input_is_not_changed(self):
        schema = {'type': 'object', 'properties': {'a': {'$ref': '#/definitions/b'}},
                  'definitions': {'b': {'type': 'integer'}}}
        before = json.dumps(schema)
        self.assertIsNotNone(grammar_api.from_schema(schema))
        self.assertEqual(json.dumps(schema), before)

    @unittest.skipUnless(os.environ.get('STRATA_GRAMMAR_TEST'), 'set STRATA_GRAMMAR_TEST to the built grammar_test')
    def test_the_engine_parses_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, g in self.grammars().items():
                path = Path(tmp) / f'{name}.gbnf'
                path.write_text(g)
                paths.append(str(path))
            run = subprocess.run([os.environ['STRATA_GRAMMAR_TEST'], '--parse', *paths], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual(run.stdout.count('ok '), len(paths), run.stdout)


class Http(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = Recording(self.tok, 'ok', max_context=4096)
        self.start()

    def start(self):
        service = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
        self.httpd = serve(service, port=0)
        self.addCleanup(self.stop)

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, body, path='/v1/chat/completions'):
        req = urllib.request.Request(f'http://127.0.0.1:{self.httpd.server_address[1]}{path}',
                                     data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.code, json.loads(response.read() or b'null') if not body.get('stream') else None
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    base = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 4}

    def test_goes_to_the_engine_with_thinking_off(self):
        code, _ = self.post({**self.base, 'grammar': GBNF})
        self.assertEqual(code, 200)
        self.assertIsInstance(self.engine.sampling['grammar'], grammar_api.Checked)
        with_grammar = list(self.engine.last_prompt)
        code, _ = self.post({**self.base, 'chat_template_kwargs': {'enable_thinking': False}})
        self.assertEqual(code, 200)
        self.assertEqual(with_grammar, self.engine.last_prompt)
        self.assertEqual(StrataEngine.sampling_keys(self.engine.sampling), StrataEngine.sampling_keys({}))

    def test_no_field_no_key(self):
        code, _ = self.post(self.base)
        self.assertEqual(code, 200)
        self.assertNotIn('grammar', self.engine.sampling)
        self.assertNotIn(' grammar=', StrataEngine.sampling_keys(self.engine.sampling))

    def test_refused_before_a_stream(self):
        cases = [
            ({'grammar': GBNF, 'tools': [{'type': 'function', 'function': {'name': 'f', 'parameters': {}}}]}, 'tools'),
            ({'grammar': GBNF, 'reasoning_effort': 'high'}, 'thinking'),
            ({'grammar': GBNF, 'response_format': {'type': 'json_object'}}, 'response_format'),
            ({'grammar': GBNF, 'continue_final_message': True,
              'messages': [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'Sure'}]}, 'prefill'),
            ({'grammar': 5}, 'grammar'),
        ]
        for extra, word in cases:
            for stream in (False, True):
                with self.subTest(word=word, stream=stream):
                    self.engine.sampling = None
                    code, body = self.post({**self.base, 'stream': stream, **extra})
                    self.assertEqual(code, 400)
                    self.assertIn(word, body['error']['message'])
                    self.assertIsNone(self.engine.sampling)      # nothing was sent to the engine

    def test_engine_without_grammars(self):
        self.engine.info = {}
        code, body = self.post({**self.base, 'grammar': GBNF})
        self.assertEqual(code, 400)
        self.assertIn('does not support grammars', body['error']['message'])
        code, _ = self.post(self.base)
        self.assertEqual(code, 200)

    def test_other_routes_ignore_the_field(self):
        code, _ = self.post({**self.base, 'grammar': GBNF}, '/v1/messages')
        self.assertEqual(code, 200)
        self.assertNotIsInstance(self.engine.sampling.get('grammar'), grammar_api.Checked)
        self.assertEqual(StrataEngine.sampling_keys(self.engine.sampling), StrataEngine.sampling_keys({}))


class Format(unittest.TestCase):
    """response_format JSON by grammar: the server option (constrained_decoding), off by default."""

    schema = {'type': 'object', 'properties': {'a': {'type': 'integer'}}, 'required': ['a'],
              'additionalProperties': False}
    base = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 16,
            'chat_template_kwargs': {'enable_thinking': False},
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'x', 'strict': True, 'schema': schema}}}

    script = '{"a":1}'

    def start(self, option, info=None):
        tok = self.tok = ByteTokenizer()
        self.engine = Recording(tok, self.script, max_context=4096, info=info)
        service = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
        service.constrained_decoding = option
        self.httpd = serve(service, port=0)
        self.addCleanup(lambda: (self.httpd.shutdown(), self.httpd.server_close()))

    def post(self, body):
        req = urllib.request.Request(f'http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions',
                                     data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.code, json.loads(response.read())

    def status(self):
        with urllib.request.urlopen(f'http://127.0.0.1:{self.httpd.server_address[1]}/v1/status', timeout=10) as r:
            return json.loads(r.read())['structured_output']

    def test_option_on(self):
        self.start(True)
        code, body = self.post(self.base)
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body['choices'][0]['message']['content']), {'a': 1})
        g = self.engine.sampling['grammar']
        self.assertIsInstance(g, grammar_api.Checked)
        self.assertIn('root ::= "{" space a-kv space "}"', g)
        self.assertIn(' grammar=', StrataEngine.sampling_keys(self.engine.sampling))
        self.assertIn('OUTPUT FORMAT REQUIREMENT', self.tok.decode(self.engine.last_prompt))   # still prompted

    def test_json_object(self):
        self.start(True)
        self.post({**self.base, 'response_format': {'type': 'json_object'}})
        self.assertIn('root ::= object', self.engine.sampling['grammar'])

    def test_option_off_is_as_before(self):
        self.start(False)
        self.post(self.base)
        on_prompt = list(self.engine.last_prompt)
        self.assertNotIn('grammar', self.engine.sampling)
        self.assertEqual(StrataEngine.sampling_keys(self.engine.sampling), StrataEngine.sampling_keys({}))
        self.start(True)
        self.post(self.base)
        self.assertEqual(on_prompt, self.engine.last_prompt)      # the prompt is the same either way

    def test_thinking_keeps_prompt_and_check(self):
        self.script = '</think>\n\n{"a":1}'
        self.start(True)
        body = {**self.base}
        body.pop('chat_template_kwargs')
        body['max_tokens'] = 64
        self.post(body)
        self.assertNotIn('grammar', self.engine.sampling)

    def test_engine_without_grammars(self):
        self.start(True, info={})
        code, _ = self.post(self.base)
        self.assertEqual(code, 200)
        self.assertNotIn('grammar', self.engine.sampling)

    def test_unconvertible_schema_is_prompted_and_checked(self):
        self.start(True)
        schema = self.schema                                  # the converter is made to fail on it
        with unittest.mock.patch.object(grammar_api, 'from_schema', return_value=None):
            code, _ = self.post({**self.base, 'response_format': {'type': 'json_schema', 'json_schema':
                                                                  {'name': 'x', 'schema': schema}}})
        self.assertEqual(code, 200)
        self.assertNotIn('grammar', self.engine.sampling)

    def test_status(self):
        self.start(True)
        self.assertEqual(self.status(), {'formats': ['json_object', 'json_schema'], 'method': 'grammar_and_validate',
                                         'constrained_decoding': True, 'stream_buffered': True})
        self.start(False)
        self.assertEqual(self.status()['constrained_decoding'], False)
        self.assertEqual(self.status()['method'], 'prompt_and_validate')
        self.start(True, info={})
        self.assertEqual(self.status()['constrained_decoding'], False)
        self.assertEqual(self.status()['method'], 'prompt_and_validate')


if __name__ == '__main__':
    unittest.main()
