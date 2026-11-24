"""Terminal clients for the shared Strata server."""
import sys
if sys.platform == "win32":
    # CPython 3.12's WMI helper (_wmi, behind platform.system() / uname()) takes an access violation in a worker thread when WMI refuses the
    # account (0x80041003) and the caller has already gone (DIAG-WINHOST-0216). Without _wmi, platform uses the registry and the environment.
    sys.modules["_wmi"] = None

import re
from pathlib import Path

SCHEMA_VERSION = 1

#: (PolyStrata) the name of this project's per-user folders, set here once: %APPDATA%, %LOCALAPPDATA% and Documents use APP_NAME, the XDG
#: folders (~/.config, ~/.local/share, ~/.local/state) XDG_NAME.  Not the Strata project's "Strata" / "strata": both can be installed on one
#: computer and must not read or write each other's settings, models or components.  Here, not in strata_control/paths.py (which takes them
#: from here), because the monitor bundle (tools/packaging/monitor_bundle.py) carries strata_cli alone and finds the connections with them.
APP_NAME = "PolyStrata"
XDG_NAME = "polystrata"

#: The version people are told (docs/data/release.json "product_version"; tools/release_manifest.py --check keeps the two equal).
#: `version()` below is the files' build number, which the engine-compatibility checks compare, so it stays numeric.
PRODUCT_VERSION = "0.2.20"


def version() -> str:
    """The terminal's version: the VERSION file of an archive, else the source tree's CMake project version."""
    root = Path(__file__).resolve().parents[1]
    try:
        text = (root / "VERSION").read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError:
        pass
    try:
        m = re.search(r"^\s*project\([^)]*?VERSION\s+([0-9]+(?:\.[0-9]+)+)",
                      (root / "CMakeLists.txt" if (root / "CMakeLists.txt").is_file() else root.parent / "engine" / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
        if m:
            return m.group(1)
    except OSError:
        pass
    return "unknown"
