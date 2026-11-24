"""Per-user state locations, one rule for the terminal and the host.

POSIX keeps $XDG_STATE_HOME (default ~/.local/state) under ``polystrata/``. Windows has no such place: state goes to
``%LOCALAPPDATA%\\PolyStrata``. The Strata project's own app names that project's folders.
"""

from __future__ import annotations

import os
from pathlib import Path

#: (PolyStrata) the name of this project's per-user folders (strata_cli/__init__.py sets them); setup.py's settings file comes from here too.
from strata_cli import APP_NAME, XDG_NAME  # noqa: E402,F401


def state_base(environ=None, *, windows: bool | None = None, home: Path | None = None) -> Path:
    """The folder that holds PolyStrata's per-user state: runtime/, tuning/, time-skip/."""
    environ = os.environ if environ is None else environ
    windows = os.name == "nt" if windows is None else windows
    home = Path.home() if home is None else home
    if windows:
        local = environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
        return Path(local) / APP_NAME
    return Path(environ.get("XDG_STATE_HOME") or str(home / ".local" / "state")) / XDG_NAME
