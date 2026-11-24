"""Which files of serve/web the server hands out below /web/locales/: the page's translations.

The flat files (app.js, app.css, ...) are served by serve/server.py itself. The locales folder has its own whitelist: a
name pattern and one type. Nothing outside it is ever read, and a name with a backslash, a colon or a percent sign is refused.
"""
from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"

# /web/locales/<lang>.json   e.g. de.json, pt-BR.json
LOCALE = re.compile(r"^/web/locales/(?P<name>[a-z]{2,3}(?:-[A-Z]{2})?\.json)$")


def subtree_file(path: str, web: Path = WEB):
    """(file, content type) for a /web/locales path, or None when it is not served."""
    if any(c in path for c in ("\\", ":", "%", "\0")) or ".." in path:
        return None
    m = LOCALE.match(path)
    if m:
        f = web / "locales" / m.group("name")
        return (f, "application/json; charset=utf-8") if f.is_file() else None
    return None
