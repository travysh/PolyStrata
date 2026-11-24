"""Pass 22: function-level coding problems, written for this test (2026-11-24), with hidden tests.

Each problem has a prompt (the spec and a few examples), a Python reference, and a seeded generator of test inputs.
The expected answers come from the reference, so they are computed, never typed.  A C++ problem has a C++ reference
too (for the self-test) and a driver that feeds it the inputs.  Run inside the image (no network):

    python func_tasks.py selftest          # every reference passes, the examples in the prompts are right
    python func_tasks.py prompts OUT       # writes OUT/<id>/task.json like build_repo.py does

grade(id, answer) is what grade.py calls."""
import json, random, re, subprocess, sys, tempfile
from pathlib import Path

TASKS = {}
PY_FORMAT = ("Answer with the complete code in one ```python block: the function(s) asked for, with the imports and "
             "helpers they need. Do not read input, print, or include tests.")
CPP_FORMAT = ("Answer with one ```cpp block holding the function and the #include lines it needs (C++17, g++). "
              "No main(), no tests.")


def task(id, lang, title, spec, examples, ref, cases, entry=None, wrapper="", cpp_ref="", driver="", fmt_in=None,
         fmt_out=None, timeout=30, banned=None, render=None):
    TASKS[id] = dict(id=id, lang=lang, title=title, spec=spec.strip(), examples=examples, ref=ref, cases=cases,
                     entry=entry, wrapper=wrapper, cpp_ref=cpp_ref, driver=driver, fmt_in=fmt_in, fmt_out=fmt_out,
                     timeout=timeout, banned=banned, render=render)


def prompt_of(t):
    show = t["render"] or (lambda args, want: f"{t['entry']}({', '.join(map(repr, args))})  ->  "
                           + (f"raises {want.__name__}" if isinstance(want, type) else repr(want)))
    ex = "\n".join("    " + show(a, w) for a, w in t["examples"])
    lang = "Python" if t["lang"] == "py" else "C++"
    return (f"Write this {lang} code.\n\n{t['spec']}\n\nExamples:\n\n{ex}\n\n"
            + (PY_FORMAT if t["lang"] == "py" else CPP_FORMAT))


# ======================================================================== Python problems

task("p01", "py", "durations", entry="parse_duration", spec="""
`parse_duration(s: str) -> int` returns the number of seconds in a duration such as "1h30m" or "2d 4h".

- A duration is one or more parts. A part is an unsigned decimal integer (ASCII digits, leading zeros allowed)
  followed directly by a unit: `d` (86400 s), `h` (3600 s), `m` (60 s) or `s` (1 s). No space between number and unit.
- The parts that are present appear in the order d, h, m, s, each at most once.
- Parts may be written together or separated by spaces or tabs; spaces and tabs may also lead and trail.
- Anything else (an empty string, an unknown or upper-case unit, a sign, a decimal point, units out of order, a
  repeated unit, a number with no unit, ...) raises ValueError.""",
     examples=[(("1h30m",), 5400), ((" 2d\t4h ",), 187200), (("90s",), 90), (("30m1h",), ValueError), (("5",), ValueError)],
     ref='''
import re
_DUR = re.compile(r"[ \\t]*(?:([0-9]+)d)?[ \\t]*(?:([0-9]+)h)?[ \\t]*(?:([0-9]+)m)?[ \\t]*(?:([0-9]+)s)?[ \\t]*")
def parse_duration(s):
    m = _DUR.fullmatch(s)
    if not m or not any(g is not None for g in m.groups()):
        raise ValueError(s)
    d, h, mi, se = (int(g) if g else 0 for g in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se
''', cases=lambda r: [(_dur(r),) for _ in range(300)] + [("",), (" ",), ("h",), ("0s",), ("007m",), ("1d2h3m4s",),
                                                          ("1 h",), ("1H",), ("-1s",), ("1.5h",), ("1h1h",), ("5ms",)])


def _dur(r):
    units = [u for u in "dhms" if r.random() < 0.5] or [r.choice("dhms")]
    parts = [str(r.randint(0, 999)).zfill(r.choice([1, 1, 1, 3])) + u for u in units]
    bad = r.random()
    if bad < 0.08 and len(parts) > 1:
        parts[0], parts[1] = parts[1], parts[0]
    elif bad < 0.14:
        parts.append(parts[-1])
    elif bad < 0.2:
        parts[-1] = parts[-1][:-1] + " " + parts[-1][-1]
    elif bad < 0.25:
        parts[-1] = parts[-1].upper()
    elif bad < 0.3:
        parts.append(str(r.randint(1, 9)))
    elif bad < 0.34:
        parts[0] = r.choice(["-", "+", "1.", "x"]) + parts[0]
    seps = lambda: r.choice(["", "", " ", "\t", "  "])
    return seps() + "".join(p + seps() for p in parts)


task("p02", "py", "semver order", entry="sort_versions", spec="""
`sort_versions(versions: list[str]) -> list[str]` returns the versions sorted from lowest to highest precedence
following Semantic Versioning 2.0.0. Every input is a valid version `MAJOR.MINOR.PATCH`, optionally followed by
`-PRERELEASE` and/or `+BUILD`.

- Compare MAJOR, MINOR, PATCH numerically, in that order.
- A version with a pre-release has lower precedence than the same version without one.
- Pre-releases are compared identifier by identifier (split on `.`): identifiers of digits only compare numerically;
  others compare by ASCII order; a numeric identifier is lower than a non-numeric one; if all compared identifiers are
  equal, the one with more identifiers is higher.
- BUILD metadata is ignored for precedence. Versions of equal precedence keep their input order.""",
     examples=[((["1.0.0", "1.0.0-alpha", "0.9.12", "1.0.0-alpha.1"],), ["0.9.12", "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0"]),
               ((["1.0.0-beta.11", "1.0.0-beta.2", "1.0.0-rc.1+b5", "1.0.0-rc.1"],),
                ["1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1+b5", "1.0.0-rc.1"])],
     ref='''
def _key(v):
    core = v.partition("+")[0]
    main, dash, pre = core.partition("-")
    nums = tuple(int(x) for x in main.split("."))
    if not dash:
        return (nums, 1, ())
    return (nums, 0, tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in pre.split(".")))
def sort_versions(versions):
    return sorted(versions, key=_key)
''', cases=lambda r: [([_semver(r) for _ in range(r.randint(0, 25))],) for _ in range(150)])

_PRE = ["alpha", "beta", "rc", "1", "2", "10", "0", "alpha.1", "alpha.beta", "beta.2", "beta.11", "rc.1", "x-y", "A",
        "alpha.10", "alpha.9", "1.alpha", "2.0", "a1", "Z"]


def _semver(r):
    v = ".".join(str(r.choice([0, 1, 1, 2, 10, 11])) for _ in range(3))
    if r.random() < 0.6:
        v += "-" + r.choice(_PRE)
    if r.random() < 0.25:
        v += "+" + r.choice(["b1", "001", "exp.sha.5114f85", "20261124"])
    return v


task("p03", "py", "TTL cache", entry="_drive", spec="""
Write a class `TTLCache(capacity: int, ttl: int)`: a least-recently-used cache whose entries also expire.
capacity >= 1 and ttl >= 1. Every method receives the current time `now` (an int); times never go backwards.

- An entry written at time t is live while now < t + ttl. Expired entries are gone: get returns None for them and they
  never count toward the capacity.
- `get(key, now)` returns the value of a live entry and makes it the most recently used, else None.
- `put(key, value, now)` stores the value with a new expiry time and makes the key the most recently used. When the key
  is new and the cache already holds `capacity` live entries, the least recently used live entry is evicted first.
- `size(now) -> int` returns the number of live entries.
Values are never None.""",
     examples=[((2, 10, [("put", "a", 1, 0), ("put", "b", 2, 1), ("get", "a", 2), ("put", "c", 3, 3), ("get", "b", 4),
                         ("get", "a", 9), ("get", "a", 10), ("size", 10)]), [None, None, 1, None, None, 1, None, 1])],
     render=lambda a, w: ("c = TTLCache(2, 10); c.put('a', 1, 0); c.put('b', 2, 1)\n    c.get('a', 2) -> 1;  "
                          "c.put('c', 3, 3)   # evicts 'b', the least recently used\n    c.get('b', 4) -> None;  "
                          "c.get('a', 9) -> 1;  c.get('a', 10) -> None   # 'a' was written at 0\n    c.size(10) -> 1"),
     wrapper='''
def _drive(capacity, ttl, ops):
    c, out = TTLCache(capacity, ttl), []
    for op in ops:
        if op[0] == "put":
            c.put(op[1], op[2], op[3]); out.append(None)
        elif op[0] == "get":
            out.append(c.get(op[1], op[2]))
        else:
            out.append(c.size(op[1]))
    return out
''', ref='''
from collections import OrderedDict
class TTLCache:
    def __init__(self, capacity, ttl):
        self.cap, self.ttl, self.d = capacity, ttl, OrderedDict()
    def _purge(self, now):
        for k in [k for k, (v, exp) in self.d.items() if exp <= now]:
            del self.d[k]
    def get(self, key, now):
        self._purge(now)
        if key not in self.d:
            return None
        self.d.move_to_end(key)
        return self.d[key][0]
    def put(self, key, value, now):
        self._purge(now)
        if key in self.d:
            del self.d[key]
        elif len(self.d) >= self.cap:
            self.d.popitem(last=False)
        self.d[key] = (value, now + self.ttl)
    def size(self, now):
        self._purge(now)
        return len(self.d)
''', cases=lambda r: [_ttl_case(r) for _ in range(120)])


def _ttl_case(r):
    now, ops = 0, []
    keys = "abcdefg"[:r.randint(2, 7)]
    for _ in range(r.randint(5, 80)):
        now += r.choice([0, 0, 1, 1, 2, 5, 13])
        x = r.random()
        if x < 0.45:
            ops.append(("put", r.choice(keys), r.randint(0, 99), now))
        elif x < 0.9:
            ops.append(("get", r.choice(keys), now))
        else:
            ops.append(("size", now))
    return (r.randint(1, 4), r.choice([1, 3, 10, 30]), ops)


task("p04", "py", "shell words", entry="shell_split", spec="""
`shell_split(s: str) -> list[str]` splits a command line into words with these rules (and only these):

- Words are separated by one or more unquoted spaces or tabs. Leading and trailing spaces and tabs make no words.
- Outside quotes, a backslash makes the next character literal, whatever it is. A backslash at the end of the input
  raises ValueError.
- Single quotes: everything up to the next single quote is literal (backslashes included).
- Double quotes: everything up to the next unescaped double quote is literal, except that `\\"` gives `"` and `\\\\`
  gives `\\`. A backslash before any other character stays, together with that character.
- Quoted and unquoted parts next to each other join into one word: `a"b c"d` is the word `ab cd`. A pair of empty
  quotes standing alone is an empty word.
- An unclosed quote raises ValueError. No other character is special.""",
     examples=[(("ls -l 'my file'",), ["ls", "-l", "my file"]), (('echo "a \\"b\\" \\$x" \\ c',), ["echo", 'a "b" \\$x', " c"]),
               (("x '' y",), ["x", "", "y"]), (("'oops",), ValueError)],
     ref='''
def shell_split(s):
    words, cur, have, i, n = [], [], False, 0, len(s)
    while i < n:
        c = s[i]
        if c in " \\t":
            if have:
                words.append("".join(cur)); cur, have = [], False
            i += 1
        elif c == "\\\\":
            if i + 1 >= n:
                raise ValueError("trailing backslash")
            cur.append(s[i + 1]); have = True; i += 2
        elif c == "'":
            j = s.find("'", i + 1)
            if j < 0:
                raise ValueError("unclosed quote")
            cur.append(s[i + 1:j]); have = True; i = j + 1
        elif c == '"':
            i += 1; have = True
            while True:
                if i >= n:
                    raise ValueError("unclosed quote")
                c = s[i]
                if c == '"':
                    i += 1
                    break
                if c == "\\\\" and i + 1 < n and s[i + 1] in '"\\\\':
                    cur.append(s[i + 1]); i += 2
                else:
                    cur.append(c); i += 1
        else:
            cur.append(c); have = True; i += 1
    if have:
        words.append("".join(cur))
    return words
''', cases=lambda r: [("".join(r.choice("ab  '\"\\\\$#\t-") for _ in range(r.randint(0, 16))),) for _ in range(500)])


task("p05", "py", "integer expressions", entry="eval_expr", banned=r"(?<![\w.])(eval|exec|compile)\s*\(|\bimport\s+ast\b|\bfrom\s+ast\b|__import__",
     spec="""
`eval_expr(s: str) -> int` evaluates an integer expression. Do not use eval, exec, compile or the ast module.

- Tokens: non-negative integer literals (ASCII digits, leading zeros allowed), `+ - * / % **`, `(` and `)`.
  Spaces and tabs between tokens are ignored.
- The grammar and the meaning are Python's for integers: `**` binds tightest and is right-associative, and its right
  operand may start with a unary sign (`2**-1`); unary `-` and `+` bind looser than `**` (so `-2**2` is -4) and tighter
  than `* / %`; `* / %` are left-associative and bind tighter than binary `+ -`.
- `/` is floor division (Python `//`); `%` is Python's modulo. Dividing by zero with either raises ZeroDivisionError.
  A negative exponent raises ValueError.
- The whole input must be one valid expression; otherwise raise ValueError. Check the syntax of the whole input before
  evaluating anything; evaluate operands left to right and raise at the first error.""",
     examples=[(("1 + 2*3",), 7), (("-2**2",), -4), (("2**3**2",), 512), (("7 / -2",), -4), (("(1+2",), ValueError),
               (("5 % 0",), ZeroDivisionError), (("2**-1",), ValueError)],
     ref='''
import re
_TOK = re.compile(r"[0-9]+|\\*\\*|[-+*/%()]")
def eval_expr(s):
    toks, i = [], 0
    while i < len(s):
        if s[i] in " \\t":
            i += 1; continue
        m = _TOK.match(s, i)
        if not m:
            raise ValueError(s)
        toks.append(m.group()); i = m.end()
    pos = 0
    def peek():
        return toks[pos] if pos < len(toks) else None
    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]
    def expr():
        node = term()
        while peek() in ("+", "-"):
            node = (take(), node, term())
        return node
    def term():
        node = factor()
        while peek() in ("*", "/", "%"):
            node = (take(), node, factor())
        return node
    def factor():
        if peek() in ("+", "-"):
            return ("u" + take(), factor())
        return power()
    def power():
        node = atom()
        if peek() == "**":
            take()
            node = ("**", node, factor())
        return node
    def atom():
        t = peek()
        if t is None:
            raise ValueError("end")
        if t == "(":
            take()
            node = expr()
            if peek() != ")":
                raise ValueError("paren")
            take()
            return node
        if t.isdigit():
            return int(take())
        raise ValueError(t)
    tree = expr()
    if pos != len(toks):
        raise ValueError("trailing")
    def ev(n):
        if isinstance(n, int):
            return n
        if n[0] in ("u-", "u+"):
            v = ev(n[1])
            return -v if n[0] == "u-" else v
        a = ev(n[1]); b = ev(n[2])
        op = n[0]
        if op == "+": return a + b
        if op == "-": return a - b
        if op == "*": return a * b
        if op == "/": return a // b
        if op == "%": return a % b
        if b < 0:
            raise ValueError("negative exponent")
        return a ** b
    return ev(tree)
''', cases=lambda r: [(_expr_case(r),) for _ in range(400)])


def _expr_gen(r, depth):
    if depth <= 0 or r.random() < 0.3:
        return str(r.randint(0, 30))
    x = r.random()
    if x < 0.12:
        return r.choice("-+") + _expr_gen(r, depth - 1)
    if x < 0.25:
        return "(" + _expr_gen(r, depth - 1) + ")"
    if x < 0.33:
        base = str(r.randint(0, 9)) if r.random() < 0.7 else "(" + _expr_gen(r, 1) + ")"
        return base + "**" + r.choice(["", "", "", "-", "+"]) + str(r.randint(0, 4))
    op = r.choice(["+", "-", "*", "/", "%", "*", "+"])
    return _expr_gen(r, depth - 1) + op + _expr_gen(r, depth - 1)


def _expr_case(r):
    s = _expr_gen(r, r.randint(1, 5))
    if r.random() < 0.2:
        i = r.randrange(len(s) + 1)
        s = s[:i] + r.choice(["(", ")", "*", "+", "", "x", " 1"]) + s[i + 1:]
    return "".join(c + (r.choice([" ", "", "", "\t"]) if r.random() < 0.3 else "") for c in s)


task("p06", "py", "token bucket", entry="admit", spec="""
`admit(times: list[int], capacity: int, period: int) -> list[bool]` simulates a token-bucket rate limiter and says,
for each request, whether it is admitted.

- The bucket starts full, with `capacity` tokens, at time 0.
- At every time that is a positive multiple of `period` (period, 2*period, ...) one token is added, unless the bucket is
  already full.
- A request at time t first receives the tokens due at times <= t, then takes one token if there is one (admitted) or
  is rejected.
- `times` is sorted (non-decreasing) and every time is >= 0. capacity >= 1, period >= 1.""",
     examples=[(([0, 0, 0, 5, 10, 10, 11], 2, 10), [True, True, False, False, True, False, False])],
     ref='''
def admit(times, capacity, period):
    tokens, last, out = capacity, 0, []
    for t in times:
        ticks = t // period
        tokens = min(capacity, tokens + ticks - last)
        last = ticks
        if tokens > 0:
            tokens -= 1
            out.append(True)
        else:
            out.append(False)
    return out
''', cases=lambda r: [_bucket(r) for _ in range(200)])


def _bucket(r):
    t, times = 0, []
    for _ in range(r.randint(0, 60)):
        t += r.choice([0, 0, 1, 2, 3, 7, 10, 25])
        times.append(t)
    return (times, r.randint(1, 5), r.choice([1, 2, 5, 10, 7]))


task("p07", "py", "JSON pointer", entry="json_pointer_get", spec="""
`json_pointer_get(doc, pointer: str)` returns the value a JSON Pointer (RFC 6901) names in `doc`, a structure of
dicts, lists, strings, numbers, booleans and None.

- The empty pointer "" names the whole document. Otherwise the pointer must start with "/" and is split on "/" into
  reference tokens. In a token, "~1" stands for "/" and "~0" for "~" (so "~01" is "~1"); a "~" followed by anything
  else, or at the end of a token, makes the pointer malformed.
- A malformed pointer raises ValueError, checked for the whole pointer before any lookup.
- In a dict the token is a key; a missing key raises KeyError.
- In a list the token must be "0" or digits with no leading zero, and in range; otherwise (including "-") raise
  IndexError.
- Stepping into anything that is not a dict or a list raises TypeError.""",
     examples=[(({"a": {"b/c": [10, 20]}}, "/a/b~1c/1"), 20), (({"~x": 1}, "/~0x"), 1), (({"a": [1]}, "/a/01"), IndexError),
               (({"a": 1}, "/a/b"), TypeError), (({"a": 1}, "/a~2"), ValueError), (({"a": 1}, ""), {"a": 1})],
     ref='''
import re
def json_pointer_get(doc, pointer):
    if pointer == "":
        return doc
    if not pointer.startswith("/"):
        raise ValueError(pointer)
    raw = pointer[1:].split("/")
    for t in raw:
        if re.search(r"~(?![01])", t):
            raise ValueError(t)
    cur = doc
    for t in raw:
        tok = t.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict):
            if tok not in cur:
                raise KeyError(tok)
            cur = cur[tok]
        elif isinstance(cur, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", tok) or int(tok) >= len(cur):
                raise IndexError(tok)
            cur = cur[int(tok)]
        else:
            raise TypeError(tok)
    return cur
''', cases=lambda r: [_ptr_case(r) for _ in range(300)])

_KEYS = ["a", "b", "", "a/b", "~", "~1", "x~0y", "0", "10", " ", "c"]


def _doc(r, depth):
    x = r.random()
    if depth <= 0 or x < 0.25:
        return r.choice([1, "s", None, True, 2.5, 0])
    if x < 0.6:
        return {k: _doc(r, depth - 1) for k in r.sample(_KEYS, r.randint(0, 4))}
    return [_doc(r, depth - 1) for _ in range(r.randint(0, 12))]


def _ptr_case(r):
    doc = _doc(r, 4)
    cur, toks = doc, []
    for _ in range(r.randint(0, 4)):
        if isinstance(cur, dict) and cur and r.random() < 0.85:
            k = r.choice(list(cur))
            toks.append(k.replace("~", "~0").replace("/", "~1"))
            cur = cur[k]
        elif isinstance(cur, list) and cur and r.random() < 0.85:
            i = r.randrange(len(cur))
            toks.append(str(i))
            cur = cur[i]
        else:
            toks.append(r.choice(["zz", "0", "01", "-", "99", "~1", "a"]))
            break
    p = "".join("/" + t for t in toks)
    if r.random() < 0.12:
        p = r.choice(["a", "~2", "/~", "/x~", p + "/~3", p[1:] if p else "q"])
    return (doc, p)


task("p08", "py", "justify text", entry="justify", spec="""
`justify(text: str, width: int) -> list[str]` lays out text in lines of exactly `width` characters (width >= 1).

- Words are the runs of non-whitespace characters of `text`. A word longer than `width` is first cut into pieces of
  `width` characters (the last piece may be shorter); each piece is then a word of its own.
- Fill lines greedily: a line takes as many of the next words as fit with one space between them.
- Every line except the last: with two or more words, add spaces between the words so the line is exactly `width`
  long, spreading them as evenly as possible; when they cannot be even, the gaps on the RIGHT get one space more than
  those on the left. A line with one word is that word padded with spaces on the right.
- The last line: words separated by single spaces, padded with spaces on the right to `width`.
- Text with no words gives an empty list.""",
     examples=[(("the quick brown fox jumps", 11), ["the   quick", "brown   fox", "jumps      "]),
               (("a b c d e", 8), ["a b c  d", "e       "]),
               (("abcdefgh xy", 3), ["abc", "def", "gh ", "xy "])],
     ref='''
def justify(text, width):
    words = []
    for w in text.split():
        while len(w) > width:
            words.append(w[:width]); w = w[width:]
        words.append(w)
    lines, cur = [], []
    for w in words:
        if cur and len(" ".join(cur + [w])) > width:
            lines.append(cur); cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    out = []
    for n, ws in enumerate(lines):
        if n == len(lines) - 1 or len(ws) == 1:
            out.append(" ".join(ws).ljust(width))
            continue
        total = width - sum(map(len, ws))
        gaps = len(ws) - 1
        base, extra = divmod(total, gaps)
        line = ws[0]
        for g in range(gaps):
            line += " " * (base + (1 if g >= gaps - extra else 0)) + ws[g + 1]
        out.append(line)
    return out
''', cases=lambda r: [(" ".join("".join(r.choice("abcdefgh") for _ in range(r.choice([1, 2, 3, 4, 5, 8, 13])))
                                for _ in range(r.randint(0, 25))) + r.choice(["", " ", "\n"]), r.randint(1, 20))
                      for _ in range(250)])


task("p09", "py", "meeting rooms", entry="assign_rooms", spec="""
`assign_rooms(meetings: list[tuple[int, int]]) -> list[int]` gives each meeting a room number.

- A meeting (start, end) takes the half-open time [start, end); start < end.
- Rooms are numbered 0, 1, 2, ... A room is free at time t when every meeting already in it has end <= t.
- Handle the meetings in order of start time (ties: lower index in the input first), and give each the lowest-numbered
  room that is free at its start.
- Return the room of each meeting, in the input's order.""",
     examples=[(([(0, 30), (5, 10), (15, 20), (10, 15)],), [0, 1, 1, 1]), (([(1, 5), (1, 5), (5, 6)],), [0, 1, 0])],
     ref='''
import heapq
def assign_rooms(meetings):
    order = sorted(range(len(meetings)), key=lambda i: (meetings[i][0], i))
    free, busy, nxt, out = [], [], 0, [0] * len(meetings)
    for i in order:
        s, e = meetings[i]
        while busy and busy[0][0] <= s:
            heapq.heappush(free, heapq.heappop(busy)[1])
        if free:
            room = heapq.heappop(free)
        else:
            room = nxt; nxt += 1
        out[i] = room
        heapq.heappush(busy, (e, room))
    return out
''', cases=lambda r: [([(s, s + r.randint(1, 12)) for s in (r.randint(0, 40) for _ in range(r.randint(0, 40)))],)
                      for _ in range(200)])


task("p10", "py", "build order", entry="build_order", spec="""
`build_order(deps: dict[str, list[str]]) -> list[str]` returns an order in which to build targets.

- `deps[t]` lists the targets t needs; they must come before t. Every name that appears anywhere (as a key or in a
  list) is a target. A list may repeat a name.
- Among the targets whose needs are all built, always take the smallest name next (ordinary string comparison).
- If the targets cannot all be ordered (a cycle, a target that needs itself), raise ValueError.""",
     examples=[(({"app": ["lib", "util"], "lib": ["util"], "test": ["app"]},), ["util", "lib", "app", "test"]),
               (({"b": [], "a": ["c"]},), ["b", "c", "a"]), (({"x": ["y"], "y": ["x"]},), ValueError)],
     ref='''
import heapq
def build_order(deps):
    nodes = set(deps) | {d for v in deps.values() for d in v}
    need = {n: set() for n in nodes}
    users = {n: set() for n in nodes}
    for t, ds in deps.items():
        for d in ds:
            need[t].add(d); users[d].add(t)
    ready = [n for n in nodes if not need[n]]
    heapq.heapify(ready)
    out = []
    while ready:
        n = heapq.heappop(ready); out.append(n)
        for u in users[n]:
            need[u].discard(n)
            if not need[u]:
                heapq.heappush(ready, u)
    if len(out) != len(nodes):
        raise ValueError("cycle")
    return out
''', cases=lambda r: [_deps(r) for _ in range(200)])


def _deps(r):
    names = [r.choice(["a", "b", "core", "lib", "x1", "x10", "x2", "util", "z", "app", "B", "test"]) + str(i % 3 or "")
             for i in range(r.randint(1, 12))]
    names = sorted(set(names))
    r.shuffle(names)
    deps = {}
    for i, n in enumerate(names):
        if r.random() < 0.8:
            pool = names[:i] if r.random() < 0.85 else names
            deps[n] = [r.choice(pool) for _ in range(r.randint(0, 3))] if pool else []
    return (deps,)


task("p11", "py", "rotate rings", entry="rotate_rings", spec="""
`rotate_rings(matrix: list[list[int]], k: int) -> list[list[int]]` rotates every ring of an m x n matrix (m, n >= 1)
clockwise by k positions and returns the new matrix (do not change the input).

- Ring 0 is the outer border, ring 1 the border of what is left inside it, and so on.
- Each ring's cells, read clockwise from its top-left corner, form a cycle; rotating by k moves every value k places
  forward along its own ring. k may be negative (counter-clockwise) and larger than the ring.
- A ring that is a single row, a single column or a single cell (the middle of some matrices) stays as it is.""",
     examples=[(([[1, 2], [3, 4]], 1), [[3, 1], [4, 2]]),
               (([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 1), [[5, 1, 2, 3], [9, 6, 7, 4], [10, 11, 12, 8]]),
               (([[1, 2, 3], [4, 5, 6], [7, 8, 9]], -1), [[2, 3, 6], [1, 5, 9], [4, 7, 8]])],
     ref='''
def rotate_rings(matrix, k):
    m, n = len(matrix), len(matrix[0])
    out = [row[:] for row in matrix]
    layer = 0
    while True:
        top, left, bottom, right = layer, layer, m - 1 - layer, n - 1 - layer
        if top >= bottom or left >= right:
            break
        pos = [(top, c) for c in range(left, right + 1)] + [(r, right) for r in range(top + 1, bottom + 1)]
        pos += [(bottom, c) for c in range(right - 1, left - 1, -1)] + [(r, left) for r in range(bottom - 1, top, -1)]
        vals = [matrix[r][c] for r, c in pos]
        L = len(pos)
        for p, v in enumerate(vals):
            r, c = pos[(p + k) % L]
            out[r][c] = v
        layer += 1
    return out
''', cases=lambda r: [_rings(r) for _ in range(200)])


def _rings(r):
    m, n = r.randint(1, 8), r.randint(1, 8)
    return ([[r.randint(0, 99) for _ in range(n)] for _ in range(m)], r.choice([0, 1, -1, 2, 3, -5, 7, 100, -123456]))


task("p12", "py", "window medians", entry="window_medians", timeout=90, spec="""
`window_medians(nums: list[int], k: int) -> list[int]` returns the median of every window of k consecutive values,
from the window starting at index 0 to the one ending at the last index (len(nums) - k + 1 values).

- The median of a window is its middle value when sorted; for an even k it is the LOWER of the two middle values.
- 1 <= k <= len(nums) <= 200,000 and |nums[i]| <= 10**9. Your function must finish within a few seconds for inputs of
  that size (for example len(nums) = 60,000 with k = 20,000).""",
     examples=[(([1, 3, -1, -3, 5, 3, 6, 7], 3), [1, -1, -1, 3, 5, 6]), (([4, 1, 3, 2], 2), [1, 1, 2]), (([5], 1), [5])],
     ref='''
def window_medians(nums, k):
    vals = sorted(set(nums))
    rank = {v: i + 1 for i, v in enumerate(vals)}
    size = len(vals)
    tree = [0] * (size + 1)
    def add(i, d):
        while i <= size:
            tree[i] += d; i += i & -i
    top = 1
    while top * 2 <= size:
        top *= 2
    def kth(q):
        pos, step = 0, top
        while step:
            if pos + step <= size and tree[pos + step] < q:
                pos += step; q -= tree[pos]
            step //= 2
        return vals[pos]
    want = (k - 1) // 2 + 1
    out = []
    for i, v in enumerate(nums):
        add(rank[v], 1)
        if i >= k:
            add(rank[nums[i - k]], -1)
        if i >= k - 1:
            out.append(kth(want))
    return out
''', cases=lambda r: [_med(r) for _ in range(120)] + [_med_big(r)])


def _med(r):
    n = r.randint(1, 40)
    return ([r.randint(-5, 5) if r.random() < 0.5 else r.randint(-10**9, 10**9) for _ in range(n)], r.randint(1, n))


def _med_big(r):
    return ([r.randint(-10**9, 10**9) for _ in range(60000)], 20000)


task("p13", "py", "business days", entry="business_days", spec="""
`business_days(start: str, end: str, holidays: list[str]) -> int` counts business days between two dates.

- Dates are "YYYY-MM-DD" (the Gregorian calendar).
- Count the days d with start <= d < end that are Monday to Friday and not in `holidays` (which may list weekends and
  repeats).
- If end is before start, return minus the count for (end, start).""",
     examples=[(("2026-11-23", "2026-11-30", []), 5), (("2026-11-23", "2026-11-30", ["2026-11-26", "2026-11-28"]), 4),
               (("2026-11-30", "2026-11-23", []), -5), (("2026-11-28", "2026-11-28", []), 0)],
     ref='''
from datetime import date, timedelta
def business_days(start, end, holidays):
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    if b < a:
        return -business_days(end, start, holidays)
    hs = {date.fromisoformat(h) for h in holidays}
    n, d = 0, a
    while d < b:
        if d.weekday() < 5 and d not in hs:
            n += 1
        d += timedelta(1)
    return n
''', cases=lambda r: [_bdays(r) for _ in range(200)])


def _bdays(r):
    import datetime
    base = datetime.date(2023, 1, 1) + datetime.timedelta(r.randint(0, 2000))
    a = base + datetime.timedelta(r.randint(0, 40) if r.random() < 0.7 else r.randint(0, 1500))
    hol = [(base + datetime.timedelta(r.randint(0, 60))).isoformat() for _ in range(r.randint(0, 6))]
    pair = [base.isoformat(), a.isoformat()]
    if r.random() < 0.3:
        pair.reverse()
    return (pair[0], pair[1], hol)


task("p14", "py", "version ranges", entry="satisfies", spec="""
`satisfies(version: str, rng: str) -> bool` says whether a version is in a version range (a subset of npm's syntax).

- A version is "X.Y.Z", three non-negative decimal integers.
- A range is one or more comparator sets separated by "||" (spaces around it are allowed). The version is in the range
  when it satisfies every comparator of at least one set. A set that is empty or only spaces matches every version.
- A set is comparators separated by spaces. The comparators:
  - `OP X.Y.Z` with OP one of >= <= > < = (no space between OP and the version), or a bare `X.Y.Z` meaning `=X.Y.Z`.
  - `^X.Y.Z`: >=X.Y.Z and below the next version that changes the first non-zero part of X, Y, Z:
    `^1.2.3` is <2.0.0, `^0.2.3` is <0.3.0, `^0.0.3` is <0.0.4 (and `^0.0.0` is <0.0.1).
  - `~X.Y.Z`: >=X.Y.Z <X.(Y+1).0.
  - X-ranges: `*`, `x` or `X` alone match anything; `1`, `1.x`, `1.*`, `1.X`, `1.x.x` mean >=1.0.0 <2.0.0; `1.2`,
    `1.2.x`, `1.2.*` mean >=1.2.0 <1.3.0.
- A hyphen range is a set of exactly the form `A - B` (spaces around the hyphen), A and B full versions: >=A <=B.
- Anything else anywhere in the range raises ValueError, even if an earlier set would match.""",
     examples=[(("1.4.0", "^1.2.3"), True), (("2.0.0", "^1.2.3"), False), (("0.2.9", "^0.2.3"), True),
               (("1.3.0", "~1.2.3"), False), (("1.5.0", "1.2.3 - 1.4.9 || 1.5"), True), (("3.0.0", ">=2.0.0 <3.0.0"), False),
               (("1.0.0", "1.0.0 || >=x"), ValueError)],
     ref='''
import re
def _ver(s):
    m = re.fullmatch(r"([0-9]+)\\.([0-9]+)\\.([0-9]+)", s)
    if not m:
        raise ValueError(s)
    return tuple(map(int, m.groups()))
def _comps(tok):
    if tok in ("*", "x", "X"):
        return []
    m = re.fullmatch(r"(>=|<=|>|<|=)?([0-9]+\\.[0-9]+\\.[0-9]+)", tok)
    if m:
        return [(m.group(1) or "=", _ver(m.group(2)))]
    m = re.fullmatch(r"([\\^~])([0-9]+\\.[0-9]+\\.[0-9]+)", tok)
    if m:
        x, y, z = _ver(m.group(2))
        if m.group(1) == "~":
            hi = (x, y + 1, 0)
        elif x:
            hi = (x + 1, 0, 0)
        elif y:
            hi = (0, y + 1, 0)
        else:
            hi = (0, 0, z + 1)
        return [(">=", (x, y, z)), ("<", hi)]
    m = re.fullmatch(r"([0-9]+)(?:\\.(?:([0-9]+)|[xX*]))?(?:\\.[xX*])?", tok)
    if m:
        x = int(m.group(1))
        if m.group(2) is None:
            return [(">=", (x, 0, 0)), ("<", (x + 1, 0, 0))]
        y = int(m.group(2))
        return [(">=", (x, y, 0)), ("<", (x, y + 1, 0))]
    raise ValueError(tok)
def _set(text):
    toks = text.split()
    if len(toks) == 3 and toks[1] == "-":
        return [(">=", _ver(toks[0])), ("<=", _ver(toks[2]))]
    out = []
    for t in toks:
        out += _comps(t)
    return out
def satisfies(version, rng):
    v = _ver(version)
    sets = [_set(part) for part in rng.split("||")]
    ok = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b,
          "=": lambda a, b: a == b}
    return any(all(ok[op](v, w) for op, w in s) for s in sets)
''', cases=lambda r: [_rng_case(r) for _ in range(400)])


def _rng_case(r):
    v = lambda: ".".join(str(r.choice([0, 0, 1, 2, 3, 10])) for _ in range(3))
    def comp():
        x = r.random()
        if x < 0.3:
            return r.choice([">=", "<=", ">", "<", "=", ""]) + v()
        if x < 0.5:
            return r.choice("^~") + v()
        if x < 0.75:
            return r.choice(["*", "x", "1", "1.x", "1.*", "0.2", "1.2.x", "2", "0", "0.0", "1.X", "1.x.x", "10.1"])
        return r.choice([">= 1.0.0", "1.2.3.4", "^1.2", "~1", "=>1.0.0", "1.x.3", "v1.0.0", "1.2.3-", "-", "a"])
    def cset():
        if r.random() < 0.15:
            return v() + " - " + v()
        return " ".join(comp() for _ in range(r.randint(0, 3)))
    rng = r.choice([" || ", "||", " ||"]).join(cset() for _ in range(r.randint(1, 3)))
    return (v(), rng)


task("p15", "py", "signed LEB128", entry="_drive", spec="""
Write two functions for signed LEB128, the variable-length integer encoding of DWARF and WebAssembly:

- `encode_sleb128(n: int) -> bytes`: the shortest signed LEB128 encoding of any Python int (any size, any sign).
- `decode_sleb128(data: bytes, offset: int = 0) -> tuple[int, int]`: decode the number that starts at `offset` and
  return (value, offset just after it). Accept any valid encoding, including one longer than needed. Raise ValueError
  if the data ends before the number does.""",
     examples=[(("enc", 0), "00"), (("enc", -1), "7f"), (("enc", 63), "3f"), (("enc", 64), "c000"), (("enc", -129), "ff7e"),
               (("dec", "02ff7e", 1), [-129, 3]), (("dec", "8080", 0), ValueError)],
     render=lambda a, w: (f"encode_sleb128({a[1]}).hex()  ->  {w!r}" if a[0] == "enc" else
                          f"decode_sleb128(bytes.fromhex({a[1]!r}), {a[2]})  ->  "
                          + (f"raises {w.__name__}" if isinstance(w, type) else repr(tuple(w)))),
     wrapper='''
def _drive(op, *a):
    if op == "enc":
        return encode_sleb128(a[0]).hex()
    v, off = decode_sleb128(bytes.fromhex(a[0]), a[1])
    return [v, off]
''', ref='''
def encode_sleb128(n):
    out = bytearray()
    while True:
        b = n & 0x7f
        n >>= 7
        if (n == 0 and not b & 0x40) or (n == -1 and b & 0x40):
            out.append(b)
            return bytes(out)
        out.append(b | 0x80)
def decode_sleb128(data, offset=0):
    result = shift = 0
    i = offset
    while True:
        if i >= len(data):
            raise ValueError("truncated")
        b = data[i]; i += 1
        result |= (b & 0x7f) << shift
        shift += 7
        if not b & 0x80:
            if b & 0x40:
                result -= 1 << shift
            return result, i
''', cases=lambda r: [_leb(r) for _ in range(400)])


def _leb(r):
    n = r.choice([r.randint(-200, 200), r.randint(-2**70, 2**70), -2**63, 2**63 - 1, 2**64, -(2**64) - 1, r.randint(-2**20, 2**20)])
    if r.random() < 0.5:
        return ("enc", n)
    enc = bytearray()
    x = n
    while True:                                          # an encoding, sometimes padded or cut short
        b = x & 0x7f; x >>= 7
        if (x == 0 and not b & 0x40) or (x == -1 and b & 0x40):
            enc.append(b); break
        enc.append(b | 0x80)
    if r.random() < 0.25:
        pad = 0x7f if n < 0 else 0x00
        enc[-1] |= 0x80
        enc += bytes([0xff if n < 0 else 0x80]) * r.randint(0, 2) + bytes([pad])
    if r.random() < 0.15:
        enc = enc[:-1]
    pre = bytes(r.randint(0, 255) for _ in range(r.randint(0, 3)))
    post = bytes(r.randint(0, 255) for _ in range(r.randint(0, 2)))
    return ("dec", (pre + bytes(enc) + post).hex(), len(pre))


task("p16", "py", "glob match", entry="glob_match", spec="""
`glob_match(pattern: str, name: str) -> bool` says whether the whole of `name` matches a glob pattern:

- `?` matches exactly one character other than "/".
- `*` matches any run of characters (possibly empty) that contains no "/".
- Two or more stars in a row (`**`) match any run of characters, "/" included.
- `[...]` matches one character, never "/", from a set: single characters and ranges like `a-z`; `[!...]` matches one
  character (other than "/") NOT in the set. A `]` right after `[` or `[!` is a member; a `-` first or last is a
  member. Inside brackets a backslash is an ordinary member. A `[` with no closing `]` is an ordinary character.
- Outside brackets, a backslash makes the next character literal; a backslash at the end matches a backslash.
- Every other character matches itself.
The check must stay fast (no exponential backtracking): patterns and names have up to 200 characters, and patterns
like `*a*a*a*a*a*a*b` against `aaaa...a` must answer at once.""",
     examples=[(("src/*.py", "src/main.py"), True), (("src/*.py", "src/a/main.py"), False), (("src/**.py", "src/a/main.py"), True),
               (("[!a-c]?", "d1"), True), (("[]x]", "]"), True), (("a\\*", "a*"), True), (("*", ""), True)],
     ref='''
def _tokens(p):
    toks, i = [], 0
    while i < len(p):
        c = p[i]
        if c == "*":
            j = i
            while j < len(p) and p[j] == "*":
                j += 1
            toks.append(("dstar",) if j - i >= 2 else ("star",)); i = j
        elif c == "?":
            toks.append(("one",)); i += 1
        elif c == "[":
            j = i + 1
            neg = j < len(p) and p[j] == "!"
            if neg:
                j += 1
            k = j + 1 if j < len(p) and p[j] == "]" else j
            close = p.find("]", k)
            if close < 0:
                toks.append(("lit", "[")); i += 1
                continue
            members, ranges, q = p[j:close], [], 0
            while q < len(members):
                if q + 2 < len(members) and members[q + 1] == "-":
                    ranges.append((members[q], members[q + 2])); q += 3
                else:
                    ranges.append((members[q], members[q])); q += 1
            toks.append(("set", neg, ranges)); i = close + 1
        elif c == "\\\\":
            toks.append(("lit", p[i + 1] if i + 1 < len(p) else "\\\\")); i += 2
        else:
            toks.append(("lit", c)); i += 1
    return toks
def _one(t, ch):
    if t[0] == "lit":
        return t[1] == ch
    if ch == "/":
        return False
    if t[0] == "one":
        return True
    inside = any(a <= ch <= b for a, b in t[2])
    return inside != t[1]
def glob_match(pattern, name):
    toks, n = _tokens(pattern), len(name)
    nxt = [False] * n + [True]
    for t in reversed(toks):
        cur = [False] * (n + 1)
        for j in range(n, -1, -1):
            if t[0] == "star":
                cur[j] = nxt[j] or (j < n and name[j] != "/" and cur[j + 1])
            elif t[0] == "dstar":
                cur[j] = nxt[j] or (j < n and cur[j + 1])
            else:
                cur[j] = j < n and _one(t, name[j]) and nxt[j + 1]
        nxt = cur
    return nxt[0]
''', cases=lambda r: [_glob_case(r) for _ in range(500)] + [("*a*a*a*a*a*a*b", "a" * 150), ("**a**a**a**a**a**b", "a/" * 90),
                                                           ("[a-c", "[a-c"), ("a\\", "a\\"), ("[\\]]", "\\]"), ("[!]]", "x")])

_GLOB_PIECES = ["a", "b", "/", "*", "**", "?", "[ab]", "[!a]", "[a-c]", "[]a]", "[-x]", "[x-]", "\\*", "\\?", ".", "[", "\\[a]"]


def _glob_case(r):
    pat = "".join(r.choice(_GLOB_PIECES) for _ in range(r.randint(0, 7)))
    if r.random() < 0.5:                                  # a name drawn from the pattern, then sometimes changed
        name = "".join(_instance(r, piece) for piece in re.findall(r"\*\*|\[!?\]?[^\]]*\]|\\.|.", pat))
        if r.random() < 0.3 and name:
            i = r.randrange(len(name))
            name = name[:i] + r.choice("ab/c.") + name[i + 1:]
    else:
        name = "".join(r.choice("ab/c.*?[]-x\\") for _ in range(r.randint(0, 8)))
    return (pat, name)


def _instance(r, piece):
    if piece == "**":
        return "".join(r.choice("ab/") for _ in range(r.randint(0, 4)))
    if piece == "*":
        return "".join(r.choice("abc") for _ in range(r.randint(0, 3)))
    if piece == "?":
        return r.choice("abcx")
    if piece.startswith("[") and len(piece) > 1:
        return r.choice("abcx-]")
    if piece.startswith("\\") and len(piece) == 2:
        return piece[1]
    return piece


# ======================================================================== C++ problems
# A driver reads the cases from stdin (count, then one case per line as fmt_in writes it) and prints one line per case.

def _cpp_driver(body):
    return "#include <bits/stdc++.h>\n#define main model_main_unused\n#include \"answer.cpp\"\n#undef main\nint main(){std::ios::sync_with_stdio(false);int T;std::cin>>T;std::string _l;std::getline(std::cin,_l);for(int t=0;t<T;++t){" + body + "}}\n"


task("c01", "cpp", "inversions", entry="count_inversions", spec="""
`long long count_inversions(const std::vector<int>& a);` returns the number of pairs i < j with a[i] > a[j].
a.size() is up to 200,000 and must be handled in well under a second.""",
     examples=[(([3, 1, 2],), 2), (([1, 2, 3],), 0), (([5, 5, 1],), 2)],
     render=lambda a, w: f"count_inversions({{{', '.join(map(str, a[0]))}}})  ->  {w}",
     ref='''
def count_inversions(a):
    def sort(v):
        if len(v) < 2:
            return v, 0
        m = len(v) // 2
        l, x = sort(v[:m]); rr, y = sort(v[m:])
        out, i, j, inv = [], 0, 0, x + y
        while i < len(l) and j < len(rr):
            if rr[j] < l[i]:
                inv += len(l) - i; out.append(rr[j]); j += 1
            else:
                out.append(l[i]); i += 1
        return out + l[i:] + rr[j:], inv
    return sort(list(a))[1]
''', cases=lambda r: [([r.randint(-5, 5) if r.random() < 0.5 else r.randint(-2**31, 2**31 - 1) for _ in range(r.randint(0, 50))],)
                      for _ in range(80)] + [([r.randint(-2**31, 2**31 - 1) for _ in range(200000)],),
                                             (list(range(200000, 0, -1)),)],
     fmt_in=lambda a: f"{len(a[0])} " + " ".join(map(str, a[0])), fmt_out=str,
     driver=_cpp_driver("int n;std::cin>>n;std::vector<int>a(n);for(auto&x:a)std::cin>>x;std::cout<<count_inversions(a)<<\"\\n\";"),
     cpp_ref=r'''
#include <vector>
#include <algorithm>
long long count_inversions(const std::vector<int>& a) {
    std::vector<int> v(a), tmp(a.size());
    long long inv = 0;
    for (size_t w = 1; w < v.size(); w *= 2) {
        for (size_t lo = 0; lo + w < v.size(); lo += 2 * w) {
            size_t mid = lo + w, hi = std::min(lo + 2 * w, v.size()), i = lo, j = mid, k = lo;
            while (i < mid && j < hi) {
                if (v[j] < v[i]) { inv += mid - i; tmp[k++] = v[j++]; } else tmp[k++] = v[i++];
            }
            while (i < mid) tmp[k++] = v[i++];
            while (j < hi) tmp[k++] = v[j++];
            for (size_t t = lo; t < hi; ++t) v[t] = tmp[t];
        }
    }
    return inv;
}
''')


task("c02", "cpp", "UTF-8 check", entry="utf8_count", spec="""
`long long utf8_count(const std::string& s);` returns the number of code points in s if s is valid UTF-8, else -1.
Valid means RFC 3629: sequences of 1 to 4 bytes with the right continuation bytes (10xxxxxx), no overlong encodings,
no surrogates (U+D800 to U+DFFF), nothing above U+10FFFF, and no sequence cut off at the end.""",
     examples=[(("68c3a9",), 2), (("e282ac",), 1), (("c080",), -1), (("eda080",), -1), (("f0908d",), -1), (("",), 0)],
     render=lambda a, w: f"utf8_count(bytes {a[0] or '(empty)'})  ->  {w}",
     ref='''
def utf8_count(hexs):
    try:
        return len(bytes.fromhex(hexs).decode("utf-8"))
    except UnicodeDecodeError:
        return -1
''', cases=lambda r: [(_utf8(r),) for _ in range(500)],
     fmt_in=lambda a: a[0] or "-", fmt_out=str,
     driver=_cpp_driver("std::string h;std::cin>>h;std::string s;if(h!=\"-\")for(size_t i=0;i<h.size();i+=2)s.push_back((char)std::stoi(h.substr(i,2),nullptr,16));std::cout<<utf8_count(s)<<\"\\n\";"),
     cpp_ref=r'''
#include <string>
long long utf8_count(const std::string& s) {
    long long n = 0; size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i]; int len; unsigned cp;
        if (c < 0x80) { len = 1; cp = c; }
        else if (c >= 0xC2 && c <= 0xDF) { len = 2; cp = c & 0x1F; }
        else if (c >= 0xE0 && c <= 0xEF) { len = 3; cp = c & 0x0F; }
        else if (c >= 0xF0 && c <= 0xF4) { len = 4; cp = c & 0x07; }
        else return -1;
        if (i + len > s.size()) return -1;
        for (int k = 1; k < len; ++k) {
            unsigned char d = s[i + k];
            if ((d & 0xC0) != 0x80) return -1;
            cp = (cp << 6) | (d & 0x3F);
        }
        if ((len == 3 && cp < 0x800) || (len == 4 && (cp < 0x10000 || cp > 0x10FFFF)) || (cp >= 0xD800 && cp <= 0xDFFF)) return -1;
        i += len; ++n;
    }
    return n;
}
''')


def _utf8(r):
    if r.random() < 0.5:
        s = "".join(chr(r.choice([r.randint(0x20, 0x7e), r.randint(0x80, 0x7ff), r.randint(0x800, 0xd7ff),
                                  r.randint(0xe000, 0xffff), r.randint(0x10000, 0x10ffff)])) for _ in range(r.randint(0, 6)))
        b = bytearray(s.encode("utf-8"))
        if b and r.random() < 0.4:
            i = r.randrange(len(b))
            b[i] = r.choice([0x80, 0xbf, 0xc0, 0xc1, 0xf5, 0xff, 0xe0, 0xed, 0x41])
        if b and r.random() < 0.15:
            b = b[:-1]
        return b.hex()
    raw = r.choice(["c080", "c1bf", "e08080", "e09f bf", "eda080", "edbfbf", "ee8080", "f08f bfbf", "f0908080", "f4908080",
                    "f48fbfbf", "f8888080", "c2", "e282", "e2", "80", "c2c2", "efbfbf", "f0", "7f", "00", "dfbf"])
    return raw.replace(" ", "")


task("c03", "cpp", "components", entry="components_after", spec="""
`std::vector<int> components_after(int n, const std::vector<std::pair<int,int>>& edges);` Nodes 0..n-1 start with no
edges. Add the undirected edges one by one in order; after each, record the number of connected components. Return
those numbers. Self-loops and repeated edges may appear. n is up to 200,000 and edges up to 300,000.""",
     examples=[((4, [(0, 1), (2, 3), (1, 0), (1, 2)]), [3, 2, 2, 1]), ((2, [(1, 1)]), [2])],
     render=lambda a, w: f"components_after({a[0]}, {{{', '.join('{%d,%d}' % e for e in a[1])}}})  ->  {{{', '.join(map(str, w))}}}",
     ref='''
def components_after(n, edges):
    p = list(range(n))
    def find(x):
        while p[x] != x:
            p[x] = p[p[x]]; x = p[x]
        return x
    comp, out = n, []
    for u, v in edges:
        a, b = find(u), find(v)
        if a != b:
            p[a] = b; comp -= 1
        out.append(comp)
    return out
''', cases=lambda r: [_comp(r, r.randint(1, 20), r.randint(0, 30)) for _ in range(100)] + [_comp(r, 200000, 300000)],
     fmt_in=lambda a: f"{a[0]} {len(a[1])} " + " ".join(f"{u} {v}" for u, v in a[1]), fmt_out=lambda w: " ".join(map(str, w)),
     driver=_cpp_driver("int n,m;std::cin>>n>>m;std::vector<std::pair<int,int>>e(m);for(auto&x:e)std::cin>>x.first>>x.second;auto o=components_after(n,e);for(size_t i=0;i<o.size();++i)std::cout<<(i?\" \":\"\")<<o[i];std::cout<<\"\\n\";"),
     cpp_ref=r'''
#include <vector>
#include <numeric>
#include <utility>
std::vector<int> components_after(int n, const std::vector<std::pair<int,int>>& edges) {
    std::vector<int> p(n); std::iota(p.begin(), p.end(), 0);
    auto find = [&](int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; };
    int comp = n; std::vector<int> out;
    for (auto [u, v] : edges) { int a = find(u), b = find(v); if (a != b) { p[a] = b; --comp; } out.push_back(comp); }
    return out;
}
''')


def _comp(r, n, m):
    return (n, [(r.randrange(n), r.randrange(n)) for _ in range(m)])


task("c04", "cpp", "sliding maximum", entry="sliding_max", spec="""
`std::vector<int> sliding_max(const std::vector<int>& a, int k);` returns the maximum of every window of k consecutive
values (a.size() - k + 1 values). 1 <= k <= a.size() <= 1,000,000; it must run in linear time.""",
     examples=[(([1, 3, -1, -3, 5, 3, 6, 7], 3), [3, 3, 5, 5, 6, 7]), (([2], 1), [2])],
     render=lambda a, w: f"sliding_max({{{', '.join(map(str, a[0]))}}}, {a[1]})  ->  {{{', '.join(map(str, w))}}}",
     ref='''
from collections import deque
def sliding_max(a, k):
    dq, out = deque(), []
    for i, v in enumerate(a):
        while dq and a[dq[-1]] <= v:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(a[dq[0]])
    return out
''', cases=lambda r: [_smax(r, r.randint(1, 40)) for _ in range(100)] + [_smax(r, 1000000, 500000), _smax(r, 1000000, 1)],
     fmt_in=lambda a: f"{len(a[0])} {a[1]} " + " ".join(map(str, a[0])), fmt_out=lambda w: " ".join(map(str, w)),
     driver=_cpp_driver("int n,k;std::cin>>n>>k;std::vector<int>a(n);for(auto&x:a)std::cin>>x;auto o=sliding_max(a,k);for(size_t i=0;i<o.size();++i)std::cout<<(i?\" \":\"\")<<o[i];std::cout<<\"\\n\";"),
     cpp_ref=r'''
#include <vector>
#include <deque>
std::vector<int> sliding_max(const std::vector<int>& a, int k) {
    std::deque<int> dq; std::vector<int> out;
    for (int i = 0; i < (int)a.size(); ++i) {
        while (!dq.empty() && a[dq.back()] <= a[i]) dq.pop_back();
        dq.push_back(i);
        if (dq.front() <= i - k) dq.pop_front();
        if (i >= k - 1) out.push_back(a[dq.front()]);
    }
    return out;
}
''')


def _smax(r, n, k=None):
    return ([r.randint(-10**9, 10**9) if r.random() < 0.7 else r.randint(-3, 3) for _ in range(n)], k or r.randint(1, n))


task("c05", "cpp", "checked expressions", entry="eval_checked", spec="""
`std::optional<long long> eval_checked(const std::string& s);` evaluates an integer expression in 64-bit signed
arithmetic, or returns std::nullopt.

- Tokens: non-negative decimal integer literals, `+`, `-`, `*`, `(`, `)`; spaces and tabs between tokens are ignored.
- Grammar: expr = term {("+" | "-") term}; term = factor {"*" factor}; factor = "-" factor | atom;
  atom = literal | "(" expr ")". Binary operators are left-associative. There is no unary plus.
- Return std::nullopt if the input is not exactly one expression, if a literal does not fit in long long, or if any
  operation's result (including unary minus) does not fit in long long.""",
     examples=[(("2 + 3*-4",), -10), (("-(9223372036854775807) - 1",), -9223372036854775808), (("9223372036854775808",), None),
               (("4611686018427387904 * 2",), None), (("+1",), None), (("(1",), None)],
     render=lambda a, w: f"eval_checked({json.dumps(a[0])})  ->  " + ("std::nullopt" if w is None else str(w)),
     ref='''
import re
def eval_checked(s):
    LO, HI = -2**63, 2**63 - 1
    toks, i = [], 0
    while i < len(s):
        if s[i] in " \\t":
            i += 1; continue
        m = re.compile(r"[0-9]+|[-+*()]").match(s, i)
        if not m:
            return None
        toks.append(m.group()); i = m.end()
    pos = 0
    class Bad(Exception):
        pass
    def fit(x):
        if not LO <= x <= HI:
            raise Bad()
        return x
    def peek():
        return toks[pos] if pos < len(toks) else None
    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]
    def expr():
        v = term()
        while peek() in ("+", "-"):
            op = take(); w = term()
            v = fit(v + w if op == "+" else v - w)
        return v
    def term():
        v = factor()
        while peek() == "*":
            take(); v = fit(v * factor())
        return v
    def factor():
        if peek() == "-":
            take()
            return fit(-factor())
        return atom()
    def atom():
        t = peek()
        if t == "(":
            take(); v = expr()
            if peek() != ")":
                raise Bad()
            take()
            return v
        if t is not None and t.isdigit():
            take()
            return fit(int(t))
        raise Bad()
    try:
        v = expr()
        if pos != len(toks):
            return None
        return v
    except Bad:
        return None
''', cases=lambda r: [(_chk_case(r),) for _ in range(400)],
     fmt_in=lambda a: "E " + a[0], fmt_out=lambda w: "none" if w is None else str(w),
     driver=_cpp_driver("std::string l;std::getline(std::cin,l);auto v=eval_checked(l.substr(2));if(v)std::cout<<*v<<\"\\n\";else std::cout<<\"none\\n\";"),
     cpp_ref=r'''
#include <optional>
#include <string>
#include <climits>
#include <cctype>
namespace {
struct Parser {
    const std::string& s; size_t i = 0; bool ok = true;
    explicit Parser(const std::string& s) : s(s) {}
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i; }
    long long fail() { ok = false; return 0; }
    long long expr() {
        long long v = term();
        while (ok) {
            ws();
            if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
                char op = s[i++]; long long r = term(); if (!ok) return 0; long long o;
                if (op == '+' ? __builtin_add_overflow(v, r, &o) : __builtin_sub_overflow(v, r, &o)) return fail();
                v = o;
            } else break;
        }
        return v;
    }
    long long term() {
        long long v = factor();
        while (ok) {
            ws();
            if (i < s.size() && s[i] == '*') { ++i; long long r = factor(); if (!ok) return 0; long long o;
                if (__builtin_mul_overflow(v, r, &o)) return fail(); v = o; }
            else break;
        }
        return v;
    }
    long long factor() {
        ws();
        if (i < s.size() && s[i] == '-') { ++i; long long v = factor(); if (!ok) return 0; if (v == LLONG_MIN) return fail(); return -v; }
        return atom();
    }
    long long atom() {
        ws();
        if (i < s.size() && s[i] == '(') { ++i; long long v = expr(); ws(); if (!ok || i >= s.size() || s[i] != ')') return fail(); ++i; return v; }
        if (i >= s.size() || !isdigit((unsigned char)s[i])) return fail();
        long long v = 0;
        while (i < s.size() && isdigit((unsigned char)s[i])) {
            int d = s[i++] - '0';
            if (__builtin_mul_overflow(v, 10, &v) || __builtin_add_overflow(v, d, &v)) return fail();
        }
        return v;
    }
};
}
std::optional<long long> eval_checked(const std::string& s) {
    Parser p(s); long long v = p.expr(); p.ws();
    if (!p.ok || p.i != s.size()) return std::nullopt;
    return v;
}
''')


def _chk_gen(r, depth):
    if depth <= 0 or r.random() < 0.3:
        return str(r.choice([r.randint(0, 20), r.randint(0, 2**63 - 1), 2**63 - 1, 2**63, 2**62, 3037000499, 3037000500]))
    x = r.random()
    if x < 0.15:
        return "-" + _chk_gen(r, depth - 1)
    if x < 0.3:
        return "(" + _chk_gen(r, depth - 1) + ")"
    return _chk_gen(r, depth - 1) + r.choice(["+", "-", "*"]) + _chk_gen(r, depth - 1)


def _chk_case(r):
    s = _chk_gen(r, r.randint(1, 4))
    if r.random() < 0.15:
        i = r.randrange(len(s) + 1)
        s = s[:i] + r.choice(["(", ")", "+", "*", "", "x", " 1", "+-"]) + s[i + 1:]
    return "".join(c + (" " if r.random() < 0.15 else "") for c in s)


task("c06", "cpp", "big integer sum", entry="add_big", spec="""
`std::string add_big(const std::string& a, const std::string& b);` adds two integers given in decimal and returns the
sum in the same form. A number is an optional "-" followed by digits with no leading zeros ("0" for zero, never
"-0"); numbers have up to 10,000 digits. The result follows the same rules.""",
     examples=[(("999", "1"), "1000"), (("-5", "3"), "-2"), (("-123", "123"), "0"), (("-99", "-1"), "-100")],
     render=lambda a, w: f"add_big({json.dumps(a[0])}, {json.dumps(a[1])})  ->  {json.dumps(w)}",
     ref='''
def add_big(a, b):
    return str(int(a) + int(b))
''', cases=lambda r: [(_big(r), _big(r)) for _ in range(400)] + [("9" * 10000, "1"), ("-" + "1" + "0" * 9999, "1")],
     fmt_in=lambda a: f"{a[0]} {a[1]}", fmt_out=str,
     driver=_cpp_driver("std::string a,b;std::cin>>a>>b;std::cout<<add_big(a,b)<<\"\\n\";"),
     cpp_ref=r'''
#include <string>
#include <algorithm>
namespace {
int cmp_mag(const std::string& x, const std::string& y) {
    if (x.size() != y.size()) return x.size() < y.size() ? -1 : 1;
    return x < y ? -1 : (x > y ? 1 : 0);
}
std::string add_mag(const std::string& x, const std::string& y) {
    std::string r; int c = 0; int i = x.size() - 1, j = y.size() - 1;
    while (i >= 0 || j >= 0 || c) { int s = c + (i >= 0 ? x[i--] - '0' : 0) + (j >= 0 ? y[j--] - '0' : 0); r.push_back('0' + s % 10); c = s / 10; }
    std::reverse(r.begin(), r.end()); return r;
}
std::string sub_mag(const std::string& x, const std::string& y) {   // x >= y
    std::string r; int b = 0; int i = x.size() - 1, j = y.size() - 1;
    while (i >= 0) { int d = (x[i--] - '0') - b - (j >= 0 ? y[j--] - '0' : 0); b = d < 0; if (d < 0) d += 10; r.push_back('0' + d); }
    while (r.size() > 1 && r.back() == '0') r.pop_back();
    std::reverse(r.begin(), r.end()); return r;
}
}
std::string add_big(const std::string& a, const std::string& b) {
    bool na = a[0] == '-', nb = b[0] == '-';
    std::string x = na ? a.substr(1) : a, y = nb ? b.substr(1) : b;
    if (na == nb) { std::string s = add_mag(x, y); return (na && s != "0" ? "-" : "") + s; }
    int c = cmp_mag(x, y);
    if (c == 0) return "0";
    if (c > 0) { std::string s = sub_mag(x, y); return (na ? "-" : "") + s; }
    std::string s = sub_mag(y, x); return (nb ? "-" : "") + s;
}
''')


def _big(r):
    x = r.random()
    if x < 0.3:
        n = r.randint(-30, 30)
    elif x < 0.6:
        n = r.randint(-10**40, 10**40)
    elif x < 0.8:
        n = r.choice([10**k for k in range(1, 30)]) * r.choice([1, -1]) + r.choice([-1, 0, 1])
    else:
        n = r.randint(-10**600, 10**600)
    return str(n)


task("c07", "cpp", "top words", entry="top_k_words", spec="""
`std::vector<std::string> top_k_words(const std::string& text, int k);` Words are the maximal runs of ASCII letters
(A-Z, a-z) in text, compared in lower case. Return the k most frequent words in lower case, more frequent first and,
for equal counts, in alphabetical order; fewer than k if there are fewer distinct words. k >= 0.""",
     examples=[(("The cat and the hat; THE end.", 2), ["the", "and"]), (("a1b2 a", 5), ["a", "b"])],
     render=lambda a, w: f"top_k_words({json.dumps(a[0])}, {a[1]})  ->  {{{', '.join(json.dumps(x) for x in w)}}}",
     ref='''
import re
from collections import Counter
def top_k_words(text, k):
    c = Counter(w.lower() for w in re.findall(r"[A-Za-z]+", text))
    return [w for w, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))][:k]
''', cases=lambda r: [("".join(r.choice(["the", "The", "a", "b", "cat", "Cat", "dog", "zebra", " ", " ", ",", "1", "-", "x", "é".encode().decode("latin-1")[0], "Q"])
                               for _ in range(r.randint(0, 60))), r.randint(0, 8)) for _ in range(200)],
     fmt_in=lambda a: f"{a[1]}\nT {a[0]}", fmt_out=lambda w: " ".join(w) if w else "-",
     driver=_cpp_driver("int k;std::cin>>k;std::getline(std::cin,_l);std::string l;std::getline(std::cin,l);auto o=top_k_words(l.substr(2),k);if(o.empty())std::cout<<\"-\";for(size_t i=0;i<o.size();++i)std::cout<<(i?\" \":\"\")<<o[i];std::cout<<\"\\n\";"),
     cpp_ref=r'''
#include <string>
#include <vector>
#include <map>
#include <algorithm>
std::vector<std::string> top_k_words(const std::string& text, int k) {
    std::map<std::string, int> c; std::string w;
    for (size_t i = 0; i <= text.size(); ++i) {
        char ch = i < text.size() ? text[i] : ' ';
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')) w.push_back(ch >= 'A' && ch <= 'Z' ? ch - 'A' + 'a' : ch);
        else if (!w.empty()) { ++c[w]; w.clear(); }
    }
    std::vector<std::pair<std::string, int>> v(c.begin(), c.end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second != b.second ? a.second > b.second : a.first < b.first; });
    std::vector<std::string> out;
    for (int i = 0; i < k && i < (int)v.size(); ++i) out.push_back(v[i].first);
    return out;
}
''')


task("c08", "cpp", "walls to break", entry="min_steps", spec="""
`int min_steps(const std::vector<std::string>& grid, int k);` The grid has one 'S' (start), one 'E' (end), open cells
'.' and walls '#'. A step moves to one of the 4 neighbouring cells inside the grid. You may enter at most k wall cells
in total (each time you enter a wall cell it counts, also the same cell again). Return the fewest steps from S to E, or
-1 if E cannot be reached. The grid is up to 60 x 60 and 0 <= k <= 20.""",
     examples=[((["S#E"], 0), -1), ((["S#E"], 1), 2), ((["S.#", "##.", "..E"], 1), 4)],
     render=lambda a, w: f"min_steps({{{', '.join(json.dumps(x) for x in a[0])}}}, {a[1]})  ->  {w}",
     ref='''
from collections import deque
def min_steps(grid, k):
    R, C = len(grid), len(grid[0])
    s = next((r, c) for r in range(R) for c in range(C) if grid[r][c] == "S")
    seen = {(s[0], s[1], 0)}
    q = deque([(s[0], s[1], 0, 0)])
    while q:
        r, c, used, d = q.popleft()
        if grid[r][c] == "E":
            return d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                nu = used + (grid[nr][nc] == "#")
                if nu <= k and (nr, nc, nu) not in seen:
                    seen.add((nr, nc, nu)); q.append((nr, nc, nu, d + 1))
    return -1
''', cases=lambda r: [_maze(r, r.randint(1, 8), r.randint(1, 8), r.randint(0, 3)) for _ in range(150)] + [_maze(r, 60, 60, 20, 0.45)],
     fmt_in=lambda a: f"{len(a[0])} {len(a[0][0])} {a[1]} " + " ".join(a[0]), fmt_out=str,
     driver=_cpp_driver("int R,C,k;std::cin>>R>>C>>k;std::vector<std::string>g(R);for(auto&x:g)std::cin>>x;std::cout<<min_steps(g,k)<<\"\\n\";"),
     cpp_ref=r'''
#include <vector>
#include <string>
#include <queue>
#include <tuple>
int min_steps(const std::vector<std::string>& grid, int k) {
    int R = grid.size(), C = grid[0].size(), sr = 0, sc = 0;
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) if (grid[r][c] == 'S') { sr = r; sc = c; }
    std::vector<int> dist(R * C * (k + 1), -1);
    auto id = [&](int r, int c, int u) { return (r * C + c) * (k + 1) + u; };
    std::queue<std::tuple<int,int,int>> q; q.push({sr, sc, 0}); dist[id(sr, sc, 0)] = 0;
    int dr[] = {1, -1, 0, 0}, dc[] = {0, 0, 1, -1};
    while (!q.empty()) {
        auto [r, c, u] = q.front(); q.pop();
        int d = dist[id(r, c, u)];
        if (grid[r][c] == 'E') return d;
        for (int t = 0; t < 4; ++t) {
            int nr = r + dr[t], nc = c + dc[t];
            if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
            int nu = u + (grid[nr][nc] == '#');
            if (nu > k || dist[id(nr, nc, nu)] >= 0) continue;
            dist[id(nr, nc, nu)] = d + 1; q.push({nr, nc, nu});
        }
    }
    return -1;
}
''')


def _maze(r, R, C, k, wall=None):
    wall = wall if wall is not None else r.choice([0.2, 0.35, 0.5, 0.7])
    g = [["#" if r.random() < wall else "." for _ in range(C)] for _ in range(R)]
    cells = [(i, j) for i in range(R) for j in range(C)]
    if len(cells) < 2:
        g = [["S", "E"]]
    else:
        a, b = r.sample(cells, 2)
        g[a[0]][a[1]] = "S"; g[b[0]][b[1]] = "E"
    return (["".join(row) for row in g], k)


# ======================================================================== running answers

RUNNER = r"""
import json, sys
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
job = json.load(open(sys.argv[1]))
ns = {"__name__": "answer"}
def norm(x):
    if isinstance(x, tuple):
        return [norm(v) for v in x]
    if isinstance(x, list):
        return [norm(v) for v in x]
    if isinstance(x, dict):
        return {k: norm(v) for k, v in x.items()}
    if isinstance(x, (bytes, bytearray)):
        return {"bytes": bytes(x).hex()}
    return x
out = []
try:
    exec(compile(job["code"] + "\n\n" + job["wrapper"], "answer.py", "exec"), ns)
    fn = ns[job["entry"]]
except BaseException as e:
    json.dump({"load_error": type(e).__name__ + ": " + str(e)[:300]}, open(sys.argv[2], "w")); sys.exit(0)
for args in job["cases"]:
    try:
        out.append(["ok", norm(fn(*args))])
    except Exception as e:
        out.append(["err", [c.__name__ for c in type(e).__mro__]])
json.dump({"out": out}, open(sys.argv[2], "w"))
"""


def extract_code(answer, lang):
    """The last fenced block in the answer's language (or any fenced block), else the whole answer."""
    tags = ("python", "py", "python3") if lang == "py" else ("cpp", "c++", "cc", "cxx")
    blocks = re.findall(r"```([^\n`]*)\n(.*?)```", answer, re.S)
    mine = [b for t, b in blocks if t.strip().lower() in tags]
    if mine:
        return mine[-1]
    if blocks:
        return max((b for _, b in blocks), key=len)
    return answer


def run_py(code, t, cases, timeout):
    with tempfile.TemporaryDirectory() as d:
        job, out = Path(d) / "job.json", Path(d) / "out.json"
        job.write_text(json.dumps({"code": code, "wrapper": t["wrapper"], "entry": t["entry"], "cases": cases}))
        try:
            subprocess.run([sys.executable, "-I", "-c", RUNNER, str(job), str(out)], cwd=d, timeout=timeout,
                           stdin=subprocess.DEVNULL, capture_output=True)
        except subprocess.TimeoutExpired:
            return {"error": f"timed out after {timeout} s"}
        if not out.exists():
            return {"error": "the code crashed the interpreter"}
        return json.loads(out.read_text())


def expected(t):
    cases = [list(c) for c in t["cases"](random.Random(t["id"]))]
    res = run_py(t["ref"], dict(t, wrapper=t["wrapper"]), cases, 600)
    assert "out" in res, (t["id"], res)
    return cases, res["out"]


def grade_py(t, code):
    cases, want = expected(t)
    got = run_py(code, t, cases, t["timeout"])
    if "out" not in got:
        return {"pass": False, "msg": got.get("load_error") or got.get("error"), "right": 0, "cases": len(cases)}
    right = 0
    first_bad = None
    for c, (wk, wv), (gk, gv) in zip(cases, want, got["out"]):
        ok = (wk == gk == "ok" and wv == gv) or (wk == gk == "err" and wv[0] in gv)
        right += ok
        if not ok and first_bad is None:
            first_bad = f"case {json.dumps(c)[:150]}: want {json.dumps([wk, wv])[:100]}, got {json.dumps([gk, gv])[:100]}"
    return {"pass": right == len(cases), "right": right, "cases": len(cases), "msg": first_bad or "all cases right"}


def grade_cpp(t, code):
    cases, want = expected(t)
    lines = [t["fmt_out"](v) for k, v in want]
    with tempfile.TemporaryDirectory() as d:
        Path(d, "answer.cpp").write_text(code)
        Path(d, "driver.cpp").write_text(t["driver"])
        c = subprocess.run(["g++", "-std=c++17", "-O2", "-o", "prog", "driver.cpp"], cwd=d, capture_output=True, text=True, timeout=120)
        if c.returncode:
            return {"pass": False, "right": 0, "cases": len(cases), "msg": "compile error: " + c.stderr[-400:]}
        stdin = f"{len(cases)}\n" + "".join(t["fmt_in"](a) + "\n" for a in cases)
        try:
            p = subprocess.run(["./prog"], cwd=d, input=stdin, capture_output=True, text=True, timeout=t["timeout"])
        except subprocess.TimeoutExpired:
            return {"pass": False, "right": 0, "cases": len(cases), "msg": f"timed out after {t['timeout']} s"}
        got = p.stdout.split("\n")
        right = sum(1 for i, w in enumerate(lines) if i < len(got) and got[i].strip() == w.strip())
        bad = next((i for i, w in enumerate(lines) if i >= len(got) or got[i].strip() != w.strip()), None)
        msg = "all cases right" if bad is None else (
            f"case {t['fmt_in'](cases[bad])[:150]!r}: want {lines[bad][:80]!r}, got {(got[bad] if bad < len(got) else '<none>')[:80]!r}"
            + (f" (exit {p.returncode})" if p.returncode else ""))
        return {"pass": right == len(cases) and p.returncode == 0, "right": right, "cases": len(cases), "msg": msg}


def grade(id, answer):
    t = TASKS[id]
    code = extract_code(answer, t["lang"])
    if t["banned"] and re.search(t["banned"], code):
        return {"pass": False, "right": 0, "cases": 0, "msg": "uses a banned function"}
    return grade_py(t, code) if t["lang"] == "py" else grade_cpp(t, code)


def selftest():
    bad = 0
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    for id, t in TASKS.items():
        ns = {}
        exec(t["ref"] + "\n" + t["wrapper"], ns)
        fn = ns[t["entry"]]
        for args, want in t["examples"]:                  # the examples in the prompt agree with the reference
            try:
                got = fn(*args)
            except Exception as e:
                got = type(e)
            if isinstance(want, type):
                ok = isinstance(got, type) and issubclass(got, want)
            else:
                ok = json.loads(json.dumps(got)) == json.loads(json.dumps(want))
            if not ok:
                bad += 1
                print(f"{id}: example {args!r}: prompt says {want!r}, reference gives {got!r}")
        answer = ("```cpp\n" + t["cpp_ref"] + "```") if t["lang"] == "cpp" else ("```python\n" + t["ref"] + "```")
        g = grade(id, answer)
        wrong = grade(id, "```python\ndef nothing(): pass\n```" if t["lang"] == "py" else "```cpp\nint nothing(){return 0;}\n```")
        cases, want = expected(t)
        kinds = {}
        for k, v in want:
            kinds[k if k == "ok" else v[0]] = kinds.get(k if k == "ok" else v[0], 0) + 1
        print(f"{id} {t['title']:22} ref {'PASS' if g['pass'] else 'FAIL'} {g['right']}/{g['cases']}  "
              f"empty answer {'fails' if not wrong['pass'] else 'PASSES?!'}  outcomes {kinds}  {g['msg'][:100]}", flush=True)
        bad += (not g["pass"]) + wrong["pass"]
    print("SELFTEST", "OK" if not bad else f"{bad} problem(s)")


def write_prompts(out):
    for id, t in TASKS.items():
        d = Path(out) / id
        d.mkdir(parents=True, exist_ok=True)
        (d / "task.json").write_text(json.dumps({"id": id, "kind": "func", "lang": t["lang"], "prompt": prompt_of(t)}, indent=1))
    print(len(TASKS), "function tasks written to", out)


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    elif sys.argv[1] == "prompts":
        write_prompts(sys.argv[2])
