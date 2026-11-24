"""A test feed (a loopback address or a folder) used once on a PC is not the feed every later update check goes to."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from strata_cli import PRODUCT_VERSION, update_cmd
from strata_control import updater


def remembered(root, source):
    state = updater.State(root)
    state.data["feed"] = {"source": source}
    state.save()


class FeedMemoryTest(unittest.TestCase):
    def source(self, stored, **args):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop(update_cmd.ENV_FEED, None)
            remembered(Path(tmp), stored)
            return update_cmd._feed_source(SimpleNamespace(feed=args.get("feed")), Path(tmp))

    def test_a_loopback_test_feed_is_forgotten(self):
        self.assertEqual(self.source("http://127.0.0.1:18831"), update_cmd.default_feed())

    def test_a_local_folder_is_forgotten(self):
        self.assertEqual(self.source(r"C:\rel\FEED-interim"), update_cmd.default_feed())
        self.assertEqual(self.source("/tmp/feed"), update_cmd.default_feed())

    def test_an_https_mirror_is_kept(self):
        self.assertEqual(self.source("https://mirror.example.com/strata-feed.json"), "https://mirror.example.com/strata-feed.json")

    def test_nothing_remembered_is_the_project_feed(self):
        self.assertEqual(self.source(None), update_cmd.default_feed())

    def test_a_named_feed_still_wins(self):
        self.assertEqual(self.source("https://a.example/f.json", feed="http://127.0.0.1:1"), "http://127.0.0.1:1")

    def test_a_remembered_project_feed_follows_the_pinned_release(self):
        # #244: a feed of the Strata project's remembered from before (its latest release, an older pin) is today's default
        for old in ("https://github.com/Niko1221/Strata/releases/latest/download/strata-feed.json",
                    "https://github.com/Niko1221/Strata/releases/download/v0.2.18/strata-feed.json"):
            self.assertEqual(self.source(old), update_cmd.default_feed())


class PinnedRelease(unittest.TestCase):
    """#244: setup and `strata update` get the Strata project's engine, Control and feed from the release this tree was merged
    from (PRODUCT_VERSION), not from its latest release, which gave the 0.2.19 tree the 0.2.20 engine."""

    def test_setup_and_update_read_the_merged_release(self):
        import setup
        self.assertRegex(update_cmd.STRATA_RELEASE, r"^v\d+(\.\d+){2,3}$")   # a tag of the Strata project's, not a PolyStrata version
        self.assertEqual(update_cmd.STRATA_RELEASE, "v" + PRODUCT_VERSION)
        folder = "https://github.com/Niko1221/Strata/releases/download/" + update_cmd.STRATA_RELEASE + "/"
        self.assertEqual(setup.PREBUILT_URL, folder)
        self.assertEqual(update_cmd.DEFAULT_FEED, folder + "strata-feed.json")
        self.assertEqual(setup.prebuilt_bases(setup.PREBUILT_URL)[-1], folder)


if __name__ == "__main__":
    unittest.main()
