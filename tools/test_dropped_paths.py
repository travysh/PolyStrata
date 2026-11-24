"""What PolyStrata leaves out of the Strata project's tree, on every merge from it: the desktop app and its host (app/) and
the Chat page of the web UI. The app people use with PolyStrata is Linger (https://github.com/simplexai-labs/linger).
A merge that brings one of these back fails here; delete it again (git rm -r) before the merge commit.

    python -m pytest tools/test_dropped_paths.py      (python tools/check --only py-tools runs it)
"""
import os
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DROPPED = (
    "app",
    "terminal/serve/web/chats-ui.js",
    "terminal/serve/web/chatstore.js",
    "terminal/serve/web/history.js",
    "terminal/serve/web/preview.js",
    "terminal/serve/web/math.js",
    "terminal/serve/web/math-adapter.js",
    "terminal/serve/web/ime.js",
    "terminal/serve/web/vendor",
    "terminal/strata_control/app_policy.py",
    "terminal/tests/snapshot/app_driver.mjs",
)


def present(paths):
    """The dropped paths in this tree: the tracked files in a git checkout, else whatever is on disk (a source zip)."""
    try:
        out = subprocess.run(["git", "-C", ROOT, "ls-files", "--", *paths], capture_output=True, text=True, check=True).stdout
        return sorted(out.split())
    except (OSError, subprocess.CalledProcessError):
        return [p for p in paths if os.path.exists(os.path.join(ROOT, p))]


class DroppedPaths(unittest.TestCase):
    def test_none_of_them_came_back(self):
        self.assertEqual(present(DROPPED), [], "dropped on every merge from the Strata project (see the top of tools/test_dropped_paths.py)")

    def test_the_web_page_loads_none_of_the_chat_scripts(self):
        with open(os.path.join(ROOT, "terminal", "serve", "web", "index.html"), encoding="utf-8") as f:
            page = f.read()
        for name in DROPPED:
            if name.startswith("terminal/serve/web/"):
                self.assertNotIn("web/" + name.rsplit("/", 1)[1], page)
        self.assertNotIn('id="view-chat"', page)


if __name__ == "__main__":
    unittest.main()
