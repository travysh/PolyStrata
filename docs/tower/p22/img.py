"""Pass 22, images: 35 generated pictures with one exact answer each, asked with thinking off and with a 2K budget.

Seven kinds, five pictures each, harder from the first to the fifth: charts, counting shapes, reading codes, tables,
code with one bug, a settings window, and where things are in a grid.  Each picture is drawn here from a fixed seed
(Pillow's own font, no font files), so the answer is known exactly and no model has seen it.
    python img.py build OUT      OUT/<id>/task.json for run_bench.py (the picture inside, as a data URL) and
                                 OUT/png/<puzzle>.png to look at.  Needs Pillow: run it in the PolyStrata image.
    python3 think.py grade OUT RUNS... / think.py score GRADED...   (the same grader and tables as the puzzles)
`build` also checks every picture: shapes do not overlap, labels are different, each code bug is the only change
between a version that fails its tests and one that passes them.
"""
import base64, io, json, random, sys
from pathlib import Path

W, H = 800, 560
COLORS = {"red": (214, 39, 40), "blue": (31, 119, 180), "green": (44, 160, 44), "orange": (255, 127, 14),
          "purple": (148, 103, 189), "black": (30, 30, 30)}
SHAPES = ["circle", "square", "triangle", "diamond", "star"]
WORDS = ["Alder", "Birch", "Cedar", "Elm", "Fir", "Hazel", "Larch", "Maple", "Oak", "Pine", "Rowan", "Spruce",
         "Willow", "Yew", "Ash", "Beech"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CODE_CHARS = "ACDEFHJKMNPRTUVWXY34679"            # none of the look-alikes 0 O Q, 1 I L, 2 Z, 5 S, 6 G, 8 B
MODES = {"off": {"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 4096},
         "2k": {"reasoning_budget_tokens": 2048, "max_tokens": 4096}}
TAIL = "\n\nLook at the picture, then end your reply with one last line of the form\nANSWER: {fmt}"


DRAWN = []                                           # every canvas made, so build() can save the last one


def canvas(w=W, h=H):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), "white")
    DRAWN.append(img)
    return img, ImageDraw.Draw(img)


def font(size):
    from PIL import ImageFont
    return ImageFont.load_default(size=size)


def text_c(d, xy, s, size, fill="black"):
    d.text(xy, s, fill=fill, font=font(size), anchor="mm")


def shape(d, kind, cx, cy, r, color):
    import math
    if kind == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif kind == "square":
        d.rectangle([cx - r * .85, cy - r * .85, cx + r * .85, cy + r * .85], fill=color)
    elif kind == "triangle":
        d.polygon([(cx, cy - r), (cx - r * .95, cy + r * .75), (cx + r * .95, cy + r * .75)], fill=color)
    elif kind == "diamond":
        d.polygon([(cx, cy - r), (cx + r * .8, cy), (cx, cy + r), (cx - r * .8, cy)], fill=color)
    else:
        pts = [(cx + (r if i % 2 == 0 else r * .45) * math.sin(i * math.pi / 5),
                cy - (r if i % 2 == 0 else r * .45) * math.cos(i * math.pi / 5)) for i in range(10)]
        d.polygon(pts, fill=color)


# ---------------------------------------------------------------- charts
def chart(rng, i):
    img, d = canvas()
    x0, y0, x1, y1 = 80, 60, 770, 470                   # plot area; values 0-100
    Y = lambda v: y1 - (y1 - y0) * v / 100
    d.line([x0, y0, x0, y1], fill="black", width=2)
    d.line([x0, y1, x1, y1], fill="black", width=2)
    for v in range(0, 101, 20):
        d.line([x0, Y(v), x1, Y(v)], fill=(225, 225, 225))
        text_c(d, (x0 - 30, Y(v)), str(v), 18)
    if i < 3:
        n = (5, 8, 12)[i]
        cats = rng.sample(WORDS, n)
        vals = rng.sample(range(8, 98), n)
        bw = (x1 - x0) / n
        for k, (c, v) in enumerate(zip(cats, vals)):
            cx = x0 + bw * (k + .5)
            d.rectangle([cx - bw * .32, Y(v), cx + bw * .32, y1], fill=COLORS["blue"])
            text_c(d, (cx, Y(v) - 14), str(v), 20 if n < 12 else 16)
            text_c(d, (cx, y1 + 22), c, 20 if n < 12 else 15)
        text_c(d, (W / 2, 25), "Trees planted per site", 24)
        if i == 0:
            c = rng.choice(cats)
            return f"What is the value shown for {c}?", "a whole number", ("int", str(vals[cats.index(c)]))
        if i == 1:
            return ("Which site has the second-highest value?", "the site's name",
                    ("text", cats[vals.index(sorted(vals)[-2])]))
        return ("What is the highest value minus the lowest value?", "a whole number",
                ("int", str(max(vals) - min(vals))))
    n = (6, 12)[i - 3]
    months = MONTHS[:n]
    step = (x1 - x0) / (n - 1)
    X = lambda k: x0 + 20 + (step - 40 / (n - 1)) * k
    if i == 3:
        vals = rng.sample(range(10, 95), n)
        d.line([(X(k), Y(v)) for k, v in enumerate(vals)], fill=COLORS["blue"], width=3)
        for k, v in enumerate(vals):
            d.ellipse([X(k) - 5, Y(v) - 5, X(k) + 5, Y(v) + 5], fill=COLORS["blue"])
            text_c(d, (X(k), Y(v) - 18), str(v), 18)
            text_c(d, (X(k), y1 + 22), months[k], 20)
        text_c(d, (W / 2, 25), "Visitors (hundreds)", 24)
        return "In which month is the value lowest?", "the month as written on the chart", \
            ("text", months[vals.index(min(vals))])
    while True:                                          # two lines, never closer than 8 where they are compared
        a = [rng.randint(10, 90) for _ in range(n)]
        b = [rng.randint(10, 90) for _ in range(n)]
        if all(abs(p - q) >= 8 for p, q in zip(a, b)):
            break
    for vals, col in ((a, "blue"), (b, "orange")):
        d.line([(X(k), Y(v)) for k, v in enumerate(vals)], fill=COLORS[col], width=3)
        for k, v in enumerate(vals):
            d.ellipse([X(k) - 5, Y(v) - 5, X(k) + 5, Y(v) + 5], fill=COLORS[col])
    for k in range(n):
        text_c(d, (X(k), y1 + 22), months[k], 18)
    d.rectangle([560, 20, 580, 34], fill=COLORS["blue"])
    d.text((588, 27), "North", fill="black", font=font(18), anchor="lm")
    d.rectangle([660, 20, 680, 34], fill=COLORS["orange"])
    d.text((688, 27), "South", fill="black", font=font(18), anchor="lm")
    text_c(d, (240, 25), "Rainfall (mm)", 24)
    return ("In how many months is North (blue) higher than South (orange)?", "a whole number",
            ("int", str(sum(p > q for p, q in zip(a, b)))))


# ---------------------------------------------------------------- counting
def place(rng, n, r, w=W, h=H, margin=10):
    pts = []
    while len(pts) < n:
        x, y = rng.uniform(r + margin, w - r - margin), rng.uniform(r + margin, h - r - margin)
        if all((x - p) ** 2 + (y - q) ** 2 > (2 * r + margin) ** 2 for p, q in pts):
            pts.append((x, y))
    return pts


def count(rng, i):
    img, d = canvas()
    n, r = (8, 14, 20, 28, 36)[i], (38, 34, 30, 26, 23)[i]
    cols, kinds = list(COLORS)[:(3, 3, 4, 4, 5)[i]], SHAPES[:(2, 3, 3, 4, 5)[i]]
    items = [(rng.choice(cols), rng.choice(kinds)) for _ in range(n)]
    for (c, k), (x, y) in zip(items, place(rng, n, r)):
        shape(d, k, x, y, r, COLORS[c])
    c, k = rng.choice(items)
    return (f"How many {c} {k}s are in the picture?", "a whole number", ("int", str(items.count((c, k)))))


# ---------------------------------------------------------------- reading
def code(rng, n):
    return "".join(rng.choice(CODE_CHARS) for _ in range(n))


def read(rng, i):
    img, d = canvas()
    if i < 3:
        s, size = code(rng, (6, 8, 10)[i]), (56, 44, 34)[i]
        d.rectangle([60, 200, W - 60, 360], outline="black", width=2)
        text_c(d, (W / 2, 280), s, size)
        return "What is the code in the box? Copy it exactly.", "the code", ("code", s)
    k, n, size = (3, 5)[i - 3], (8, 12)[i - 3], (30, 24)[i - 3]
    labels = "ABCDE"[:k]
    codes = [code(rng, n) for _ in labels]
    for j, (lab, s) in enumerate(zip(labels, codes)):
        y = 90 + j * (380 / k)
        d.text((90, y), f"{lab}:", fill="black", font=font(size), anchor="lm")
        d.text((160, y), s, fill="black", font=font(size), anchor="lm")
    j = rng.randrange(k)
    return f"What is the code on line {labels[j]}? Copy it exactly.", "the code", ("code", codes[j])


# ---------------------------------------------------------------- tables
def table(rng, i):
    rows, cols = (4, 6, 8, 10, 12)[i], (3, 4, 5, 6, 6)[i]
    names = rng.sample(WORDS, rows)
    heads = [f"Q{c + 1}" if cols <= 4 else f"W{c + 1}" for c in range(cols)]
    vals = [[rng.randint(10, 999) for _ in range(cols)] for _ in range(rows)]
    rh = min(44, 500 // (rows + 1))
    size = 20 if rows <= 8 else 17
    img, d = canvas(W, rh * (rows + 1) + 40)
    cw = (W - 200) / cols
    for r in range(rows + 1):
        y = 20 + r * rh
        if r % 2 == 0 and r:
            d.rectangle([20, y, W - 20, y + rh], fill=(242, 242, 242))
        d.line([20, y, W - 20, y], fill=(180, 180, 180))
        cells = ["Site"] + heads if r == 0 else [names[r - 1]] + [str(v) for v in vals[r - 1]]
        for c, s in enumerate(cells):
            x = 100 if c == 0 else 180 + cw * (c - .5)
            text_c(d, (x, y + rh / 2), s, size, "black")
    d.line([20, 20 + (rows + 1) * rh, W - 20, 20 + (rows + 1) * rh], fill=(180, 180, 180))
    r, c = rng.randrange(rows), rng.randrange(cols)
    if i < 2:
        return f"What is the value for {names[r]} in column {heads[c]}?", "a whole number", ("int", str(vals[r][c]))
    col = [v[c] for v in vals]
    if i == 2:
        if col.count(max(col)) > 1:
            return None
        return (f"Which site has the largest value in column {heads[c]}?", "the site's name",
                ("text", names[col.index(max(col))]))
    if i == 3:
        return f"What is the sum of the values in the row for {names[r]}?", "a whole number", ("int", str(sum(vals[r])))
    return (f"How many values in column {heads[c]} are greater than 500?", "a whole number",
            ("int", str(sum(v > 500 for v in col))))


# ---------------------------------------------------------------- code with one bug
BUGS = [  # (lines, index of the bug line, the buggy line, tests as (args, result))
    (['def sum_even(xs):', '    """The sum of the even numbers in xs."""', '    total = 0', '    for x in xs:',
      '        if x % 2 == 0:', '            total += x', '    return total'],
     4, '        if x % 2 == 1:', [(([1, 2, 3, 4],), 6), (([5, 7],), 0), (([],), 0)]),
    (['def factorial(n):', '    """n! for n >= 0 (0! is 1)."""', '    result = 1', '    for i in range(2, n + 1):',
      '        result *= i', '    return result'],
     3, '    for i in range(2, n):', [((0,), 1), ((1,), 1), ((5,), 120)]),
    (['def count_vowels(s):', '    """How many letters of s are vowels (a, e, i, o, u), in upper or lower case."""',
      '    count = 0', '    for ch in s.lower():', '        if ch in "aeiou":', '            count += 1',
      '    return count'],
     3, '    for ch in s:', [(("Apple",), 2), (("xyz",), 0), (("AEIOU",), 5)]),
    (['def average_positive(xs):', '    """The mean of the numbers in xs that are greater than 0, or 0.0 if none are."""',
      '    pos = [x for x in xs if x > 0]', '    if not pos:', '        return 0.0', '    total = sum(pos)',
      '    return total / len(pos)'],
     6, '    return total / len(xs)', [(([1, -1, 3],), 2.0), (([-2],), 0.0), (([4, 6],), 5.0)]),
    (['def merge(a, b):', '    """Merge two sorted lists into one sorted list."""', '    i = j = 0', '    out = []',
      '    while i < len(a) and j < len(b):', '        if a[i] <= b[j]:', '            out.append(a[i])',
      '            i += 1', '        else:', '            out.append(b[j])', '            j += 1',
      '    out.extend(a[i:])', '    out.extend(b[j:])', '    return out'],
     12, '    out.extend(b[i:])', [(([1, 4], [2, 3, 9]), [1, 2, 3, 4, 9]), (([], [1, 2]), [1, 2]),
                                  (([5, 6, 7], [1]), [1, 5, 6, 7])]),
]


def run_code(lines, args):
    ns = {}
    exec("\n".join(lines), ns)
    f = ns[next(l for l in lines if l.startswith("def "))[4:].split("(")[0]]
    return f(*[list(a) if isinstance(a, list) else a for a in args])


def bug(rng, i):
    lines, at, bad, tests = BUGS[i]
    pre = rng.sample(["import math", "import sys", "# helpers for the report", "from typing import List"],
                     rng.randint(0, 3))
    pre = pre + [""] if pre else pre
    good = pre + lines
    buggy = list(good)
    buggy[len(pre) + at] = bad
    assert all(run_code(good, a) == want for a, want in tests), lines[0]
    assert any(run_code(buggy, a) != want for a, want in tests), lines[0]
    lh = 30
    img, d = canvas(W, lh * len(buggy) + 40)
    d.rectangle([0, 0, 52, lh * len(buggy) + 40], fill=(240, 240, 240))
    for k, s in enumerate(buggy):
        y = 20 + k * lh + lh / 2
        d.text((40, y), str(k + 1), fill=(120, 120, 120), font=font(19), anchor="rm")
        lead = len(s) - len(s.lstrip(" "))              # the font is proportional: indent by a fixed width per space
        d.text((66 + 11 * lead, y), s.lstrip(" "), fill="black", font=font(19), anchor="lm")
    return ("This Python function has exactly one bug: one line makes it do something other than what its "
            "docstring says. Which line number is it?", "the line number", ("int", str(len(pre) + at + 1)))


# ---------------------------------------------------------------- a settings window
BUTTONS = ["Save", "Cancel", "Apply", "Reset", "Export", "Import", "Delete", "Share", "Print", "Help"]
OPTIONS = ["Dark mode", "Auto update", "Sync files", "Send reports", "Show hints", "Sound", "Beta features",
           "Backups"]
FIELDS = [("Host", lambda r: f"{r.choice(['nova', 'kite', 'mesa', 'dune'])}-{r.randint(10, 99)}.lan"),
          ("Port", lambda r: str(r.randint(1025, 65000))), ("User", lambda r: r.choice(["rosa", "ivan", "mei", "tom"])),
          ("Folder", lambda r: f"/srv/{r.choice(['data', 'media', 'logs'])}{r.randint(1, 9)}"),
          ("Timeout", lambda r: f"{r.randint(5, 120)} s")]


def ui(rng, i):
    img, d = canvas()
    d.rectangle([40, 30, W - 40, H - 30], outline=(150, 150, 150), width=2, fill=(250, 250, 250))
    d.rectangle([40, 30, W - 40, 70], fill=(225, 230, 238))
    d.text((60, 50), "Settings", fill="black", font=font(22), anchor="lm")
    if i < 2:
        n = (4, 6)[i]
        labels = rng.sample(BUTTONS, n)
        off = rng.randrange(n)
        for k, lab in enumerate(labels):
            row, col = divmod(k, 3 if n == 6 else 4)
            x, y = 90 + col * (170 if n == 4 else 210), 200 + row * 110
            grey = k == off
            d.rounded_rectangle([x, y, x + 140, y + 54], 10, fill=(232, 232, 232) if grey else COLORS["blue"],
                                outline=(205, 205, 205) if grey else None)
            text_c(d, (x + 70, y + 27), lab, 22, (170, 170, 170) if grey else "white")
        return ("One button in this window is disabled (greyed out). Which one?", "the button's label",
                ("text", labels[off]))
    if i == 2 or i == 3:
        n = (6, 8)[i - 2]
        labels = rng.sample(OPTIONS, n)
        on = sorted(rng.sample(range(n), rng.randint(2, 4)))
        for k, lab in enumerate(labels):
            y = 100 + k * (400 / n) + 10
            if i == 2:
                d.rectangle([90, y - 13, 116, y + 13], outline="black", width=2,
                            fill=COLORS["blue"] if k in on else "white")
                if k in on:
                    d.line([(96, y), (102, y + 7), (111, y - 8)], fill="white", width=3)
                d.text((132, y), lab, fill="black", font=font(22), anchor="lm")
            else:
                d.text((90, y), lab, fill="black", font=font(22), anchor="lm")
                d.rounded_rectangle([560, y - 15, 620, y + 15], 15, fill=COLORS["green"] if k in on else (200, 200, 200))
                kx = 605 if k in on else 575
                d.ellipse([kx - 12, y - 12, kx + 12, y + 12], fill="white")
        what = "checked" if i == 2 else "switched on"
        return (f"Which options are {what}? List all of them, separated by commas.", "the labels, separated by commas",
                ("set", "|".join(labels[k] for k in on)))
    vals = [(name, make(rng)) for name, make in FIELDS]
    for k, (name, v) in enumerate(vals):
        y = 120 + k * 80
        d.text((90, y), name, fill="black", font=font(22), anchor="lm")
        d.rectangle([250, y - 20, 650, y + 20], outline=(160, 160, 160), fill="white")
        d.text((262, y), v, fill="black", font=font(22), anchor="lm")
    name, v = rng.choice(vals)
    return f"What is written in the {name} field?", "the text in the field", ("code", v)


# ---------------------------------------------------------------- where things are
def space(rng, i):
    n = (3, 3, 4, 4, 5)[i]
    combos = rng.sample([(c, s) for c in COLORS for s in SHAPES], n * n)
    grid = [combos[r * n:(r + 1) * n] for r in range(n)]
    cell = min(500 // n, 140)
    img, d = canvas(cell * n + 40, cell * n + 40)
    for r in range(n):
        for c in range(n):
            col, s = grid[r][c]
            shape(d, s, 20 + c * cell + cell / 2, 20 + r * cell + cell / 2, cell * .32, COLORS[col])
    name = lambda rc: f"{grid[rc[0]][rc[1]][0]} {grid[rc[0]][rc[1]][1]}"
    if i in (0, 1, 4):
        dr, dc, word = [(0, -1, "directly to the left of"), (-1, 0, "directly above"), (1, 0, "directly below")][
            (0, 1, 4).index(i)]
        r, c = rng.choice([(r, c) for r in range(n) for c in range(n) if 0 <= r + dr < n and 0 <= c + dc < n])
        return (f"What is {word} the {name((r, c))}? Give its color and shape.", "color shape",
                ("pair", name((r + dr, c + dc))))
    r, c = rng.randrange(n), rng.randrange(n)
    if i == 2:
        return (f"In which row and column is the {name((r, c))}? Count rows from the top and columns from the left, "
                f"starting at 1.", "row R, column C", ("rc", f"{r + 1},{c + 1}"))
    col = grid[r][c][0]
    others = [rr for rr in range(n) for cc in range(n) if grid[rr][cc][0] == col and (rr, cc) != (r, c)]
    return (f"How many other {col} shapes are there besides the {name((r, c))}?", "a whole number",
            ("int", str(len(others))))


FAMILIES = {"chart": chart, "count": count, "read": read, "table": table, "bug": bug, "ui": ui, "space": space}


def build(out):
    (out / "png").mkdir(parents=True, exist_ok=True)
    for fam, make in FAMILIES.items():
        for i in range(5):
            for tries in range(100):
                rng = random.Random(f"p22-img-{fam}{i + 1}-{tries}")
                got = make(rng, i)
                if got:
                    break
            (q, fmt, (kind, ans)), img = got, DRAWN[-1]
            buf = io.BytesIO()
            img.save(buf, "PNG")
            png = buf.getvalue()
            (out / "png" / f"{fam}{i + 1}.png").write_bytes(png)
            url = "data:image/png;base64," + base64.b64encode(png).decode()
            prompt = q + TAIL.format(fmt=fmt)
            for mode, req in MODES.items():
                d = out / f"{fam}{i + 1}-{mode}"
                d.mkdir(parents=True, exist_ok=True)
                msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                    {"type": "text", "text": prompt}]}]
                (d / "task.json").write_text(json.dumps({
                    "id": d.name, "kind": "image", "puzzle": f"{fam}{i + 1}", "family": fam, "level": i + 1,
                    "mode": mode, "prompt": prompt, "answer": ans, "answer_kind": kind,
                    "request": {**req, "messages": msg}}, indent=1))
            print(f"{fam}{i + 1:<3} {kind:5} {ans:24} {img.size[0]}x{img.size[1]} {len(png) // 1024} KB  {q}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build(Path(sys.argv[2]))
    else:
        sys.exit(__doc__)
