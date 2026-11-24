"""Every tracked path stays short enough to unzip on Windows. GitHub's source zip unpacks into PolyStrata-main\\, which a
browser puts in a folder such as C:\\Users\\<name>\\Downloads\\PolyStrata-main\\: about 90 characters before the repository's
own path, and Windows stops at 260 by default. A benchmark report nested inside another report's runs/ is the usual
way past it: name the inner folder report/ (the outer folder already says whose run it is).

    python -m pytest tools/test_path_length.py      (python tools/check --only py-tools runs it)
"""
import os
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIMIT = 170


class PathLength(unittest.TestCase):
    def test_tracked_paths_fit_windows(self):
        try:
            out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                                 check=True).stdout
            paths = [p for p in out.splitlines() if p]
        except (OSError, subprocess.CalledProcessError):
            paths = [os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, "/")
                     for d, _, fs in os.walk(ROOT) if ".git" not in d.split(os.sep) for f in fs]
        long = sorted((p for p in paths if len(p) > LIMIT), key=len, reverse=True)
        self.assertEqual(long, [], "paths over %d characters (shorten the folder names): %s" % (LIMIT, long[:5]))


if __name__ == "__main__":
    unittest.main()
