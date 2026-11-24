"""Tests for setup.py's Inkling 2 Mini (docs/models/inkling.md): its three pinned files under the repository's
Mainline_Support folder, where their names end in .gguf_file, and the install with no draft model, no image encoder
and one request at a time.  Mocked like tools/test_setup_glm.py, whose harness this uses.

    python -m unittest tools.test_setup_inkling
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools import test_setup_glm as glm_tests  # noqa: E402

FAM = setup.FAMILIES["inkling"]
SIZES = [("UD-Q4_K_XL", 17629760736), ("UD-Q3_K_XL", 13631779296), ("UD-Q2_K_XL", 9138716608)]


class Pins(unittest.TestCase):
    def test_the_files(self):
        self.assertEqual(list(setup.family_models("inkling")), [m for m, _ in SIZES])
        self.assertRegex(FAM["hf"], r"^https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/[0-9a-f]{40}/"
                                    r"Mainline_Support/$")
        self.assertEqual(list(FAM["sha256"]), [f"Inkling-2-Mini-{m}.gguf" for m, _ in SIZES] + [FAM["mmproj"]])
        for size, n in SIZES:
            file = f"Inkling-2-Mini-{size}.gguf"
            d = FAM["models"][size]
            self.assertEqual((setup.model_file(FAM, size, 1), setup.model_shards(FAM, size)), (file, 1))
            self.assertEqual(FAM["sha256"][file][0], n)
            self.assertRegex(FAM["sha256"][file][1], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(n / 1e9, d["download_gb"], delta=0.05)
            self.assertLess(d["arena_gb"], d["download_gb"])
            self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)), size)

    def test_the_engine_s_name_for_its_files(self):
        # the repository's top-level files are Unsloth's layout (general.architecture inkling2), which the engine
        # reads too (inkling2.cpp): a --gguf-dir file of either is taken
        self.assertEqual(setup.POLY_GGUF_ARCHS["inkling"], ("inkling2_loop", "inkling2"))


class Main(unittest.TestCase):
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_install_each_size(self):
        for size, _n in SIZES:
            with self.subTest(size):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, _ = self.main(family="inkling", model=size, ram=64.0, vram=32607 / 1024)
                self.assertEqual(code, 0, out)
                cfg = json.loads((self.t / f"strata-inkling-{size.lower()}.json").read_text())
                file = f"Inkling-2-Mini-{size}.gguf"
                self.assertEqual(self.downloads, [FAM["hf"] + file + "_file"])     # saved without the suffix
                self.assertEqual(self.verified, [(file, *FAM["sha256"][file])])
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / f"inkling-{size}" / file)
                # a card that holds every expert runs the file as it is (small_card_args: none)
                for flag in ("--draft-model", "--tail-skip", "--prompt-batch", "--batch", "--attn-requant",
                             "--loop-bf16", "--expert-swaps"):
                    self.assertNotIn(flag, args)
                self.assertNotIn("parallel", cfg)                                 # opt-in: written when asked
                self.assertNotIn("vision", cfg)
                self.assertIn("--parallel 2 to 4", out)                           # offered, as for the other families

    def test_a_card_that_holds_part_of_the_experts_gets_the_small_card_flags(self):
        # an 8 GB card: Q5_0 attention, BF16 loop controllers, tail skip 4; expert swaps stay on (the default)
        code, out, _ = self.main(family="inkling", model="UD-Q3_K_XL", ram=32.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        args = json.loads((self.t / "strata-inkling-ud-q3_k_xl.json").read_text())["args"]
        self.assertEqual(args[args.index("--attn-requant") + 1], "q5_0")
        self.assertIn("--loop-bf16", args)
        self.assertEqual(args[args.index("--tail-skip") + 1], "4")
        self.assertNotIn("--expert-swaps", args)
        self.assertEqual(args.count("--tail-skip"), 1)
        self.assertIn("holds part of", out)
        self.assertNotIn("small_card", json.loads((self.t / "strata-inkling-ud-q3_k_xl.json").read_text()))

    CFG = "strata-inkling-ud-q3_k_xl.json"
    SMALL = ("--attn-requant", "--loop-bf16", "--tail-skip")

    def small(self, argv=(), answers=None):
        """UD-Q3_K_XL on an 8 GB card, which holds part of its experts: the run config's args and "small_card"."""
        code, out, _ = self.main(argv, family="inkling", model="UD-Q3_K_XL", ram=32.0, vram=8188 / 1024,
                                 answers=answers)
        self.assertEqual(code, 0, out)
        cfg = json.loads((self.t / self.CFG).read_text())
        return cfg["args"], cfg.get("small_card"), out

    def assert_on(self, args):
        self.assertEqual(args[args.index("--attn-requant") + 1], "q5_0")
        self.assertIn("--loop-bf16", args)
        self.assertEqual(args[args.index("--tail-skip") + 1], "4")
        self.assertEqual(args.count("--tail-skip"), 1)

    def test_small_card_default_is_on_and_not_asked(self):
        # the default stays the small-card settings; no question is added (Enter for every other question)
        args, small, _ = self.small(answers={})
        self.assertEqual([q for q in self.asked if "small" in q.lower()], [])
        self.assert_on(args)
        self.assertIsNone(small)

    def test_small_card_off_is_the_high_accuracy_mode_and_kept(self):
        args, small, out = self.small(("--inkling-small-card", "off"))
        for flag in self.SMALL:
            self.assertNotIn(flag, args)
        self.assertEqual(small, "off")
        self.assertIn("high-accuracy mode", out)
        # a later setup run without the option keeps it, with --yes and with the questions asked
        for answers in (None, {}):
            args, small, _ = self.small(answers=answers)
            for flag in self.SMALL:
                self.assertNotIn(flag, args)
            self.assertEqual(small, "off")
        # --inkling-small-card on puts them back, and the config is the default one again
        args, small, _ = self.small(("--inkling-small-card", "on"))
        self.assert_on(args)
        self.assertIsNone(small)
        args, small, _ = self.small()
        self.assert_on(args)
        self.assertIsNone(small)

    def test_small_card_off_on_a_card_that_holds_every_expert_is_kept(self):
        # nothing to leave out there, but the choice is written, so the install moved back to an 8 GB card stays off
        code, out, _ = self.main(("--inkling-small-card", "off"), family="inkling", model="UD-Q3_K_XL", ram=64.0,
                                 vram=32607 / 1024)
        self.assertEqual(code, 0, out)
        cfg = json.loads((self.t / self.CFG).read_text())
        for flag in self.SMALL:
            self.assertNotIn(flag, cfg["args"])
        self.assertEqual(cfg["small_card"], "off")
        code, out, _ = self.main(family="inkling", model="UD-Q3_K_XL", ram=64.0, vram=32607 / 1024)   # again, 32 GB
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads((self.t / self.CFG).read_text())["small_card"], "off")
        args, small, _ = self.small()                                              # then an 8 GB card
        for flag in self.SMALL:
            self.assertNotIn(flag, args)
        self.assertEqual(small, "off")

    def test_small_card_off_with_two_cards(self):
        # Inkling 2 runs on one of them (an 8 GB card here), so off applies there as on one card
        code, out, cfg = self.main(("--gpus", "0,1", "--inkling-small-card", "off"), family="inkling",
                                   model="UD-Q3_K_XL", ram=32.0, vram=8188 / 1024, n_gpus=2)
        self.assertEqual(code, 0, out)
        for flag in self.SMALL:
            self.assertNotIn(flag, cfg["args"])
        self.assertEqual(cfg["small_card"], "off")

    def test_small_card_option_of_another_family_is_not_used(self):
        code, out, _ = self.main(("--inkling-small-card", "off"))                 # GLM, the harness's family
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"not used with .*--inkling-small-card")

    def test_two_cards_set_up_one_and_size_it_for_that_card(self):
        # PR 208: the engine has no layer split for Inkling 2 yet; two 8 GB cards chosen together get the config one
        # 8 GB card gets (the small-card flags), not a pair the engine refuses
        code, out, cfg = self.main(("--gpus", "0,1"), family="inkling", model="UD-Q3_K_XL", ram=32.0, vram=8188 / 1024,
                                   n_gpus=2)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 0)
        self.assertNotIn("layer_split", cfg)
        self.assertIn("runs on one card so far", out)
        args = cfg["args"]
        self.assertEqual(args[args.index("--attn-requant") + 1], "q5_0")
        self.assertEqual(args[args.index("--tail-skip") + 1], "4")
        self.assertIn("holds part of", out)
        code, out, cfg = self.main(family="inkling", model="UD-Q3_K_XL", ram=64.0, vram=16303 / 1024, n_gpus=2)
        self.assertEqual(code, 0, out)
        self.assertIsInstance(cfg["gpu"], int)
        self.assertNotIn("layer_split", cfg)

    def test_requests_at_once_are_taken(self):
        # the engine's batch slots ("parallel"), opt-in as for the other strata-poly families
        code, out, _ = self.main(("--parallel", "4"), family="inkling", model="UD-Q3_K_XL", ram=64.0, vram=32607 / 1024)
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads((self.t / "strata-inkling-ud-q3_k_xl.json").read_text())["parallel"], 4)
        self.assertNotIn("answers one request at a time", out)
        self.assertIn("parallel requests: 4 at once", out)
        self.assertNotIn("with --tail-skip", out)                          # a card that holds every expert: no skip

    def test_requests_at_once_with_the_tail_skip_are_warned_of(self):
        # an 8 GB card's small_card_args skip tail experts over a step's rows, which are several chats' tokens
        code, out, _ = self.main(("--parallel", "2"), family="inkling", model="UD-Q3_K_XL", ram=32.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads((self.t / "strata-inkling-ud-q3_k_xl.json").read_text())["parallel"], 2)
        self.assertIn('"parallel": 2 with --tail-skip', out)
        # setup again without --parallel: the config keeps "parallel": 2, and the warning with it
        code, out, _ = self.main(family="inkling", model="UD-Q3_K_XL", ram=32.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads((self.t / "strata-inkling-ud-q3_k_xl.json").read_text())["parallel"], 2)
        self.assertIn('"parallel": 2 with --tail-skip', out)



if __name__ == "__main__":
    unittest.main()
