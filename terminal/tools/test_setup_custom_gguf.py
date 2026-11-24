"""Tests for resolve_custom_gguf (--custom-gguf PATH) in setup.py.

strata_inspect is imported inside resolve_custom_gguf via `from strata_inspect import ...`,
so it is not a top-level attribute of setup. We patch it through sys.modules, injected
before the function runs so its inline import picks up our stubs.

Run: python tools/test_setup_custom_gguf.py  (no GPU or server needed)
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import setup  # noqa: E402


def _write_gguf(path: Path) -> None:
    """Write minimal bytes so .is_file() is True."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\x00\x01\x00\x00")


class CustomGgufTests(unittest.TestCase):

    def _make_stub(self, fp_dict: dict, kind: str, table: dict | None = None):
        """Build a stub module for sys.modules['strata_inspect'].

        If table is None, use the real data/gguf_fingerprints.json. Otherwise use the given dict.
        """
        m = types.ModuleType("strata_inspect")
        m.read_headers = lambda sources: (fp_dict.get("arch", "qwen4exp"), [])
        m.fingerprint = lambda arch, tensors: fp_dict
        m.verdict = lambda fp, t: (kind, f"mock verdict: {kind}")
        m.PREVIEW_ARCHS = {}
        m.POLY_ARCHS = {}
        if table is None:
            _real_path = Path(_ROOT) / "data" / "gguf_fingerprints.json"
            m.load_fingerprints = lambda path=None: json.loads(_real_path.read_text(encoding="utf-8"))
        else:
            m.load_fingerprints = lambda path=None: table
        return m

    def test_known_verdict(self):
        """A single .gguf whose fp matches exactly one fingerprint entry → (family, model, [shard])."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "qwen36-model.gguf"
            _write_gguf(p)

            # Use the first real entry from the fingerprints file to guarantee a match
            fp_table = json.loads(
                (Path(_ROOT) / "data" / "gguf_fingerprints.json").read_text(encoding="utf-8")
            )
            known = {k: v for k, v in fp_table.items() if not k.startswith("_")}
            first_key, first_entry = next(iter(known.items()))

            stub = self._make_stub(fp_dict=first_entry, kind="known")
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                family, model, shards = setup.resolve_custom_gguf(str(p))

        fam_model = first_key.split("/")
        self.assertEqual(family, fam_model[0])
        self.assertEqual(model, fam_model[1])
        self.assertEqual(shards, [p])
        self.assertTrue(shards[0].suffix == ".gguf")

    def test_multi_match_fails(self):
        """fp matches 2+ entries in the table → SystemExit naming both candidates."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.gguf"
            _write_gguf(p)

            # Craft a table where two keys map to identical fp values → multi-match
            fake_fp = {"arch": "qwen4exp", "tensors": 100, "bytes": 1000,
                       "expert_bytes": {}, "experts_bpw": None, "bpw": None, "roles": {}}
            table = {"fam_a/m1": fake_fp, "fam_b/m2": fake_fp}

            stub = self._make_stub(fp_dict=fake_fp, kind="known", table=table)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                try:
                    setup.resolve_custom_gguf(str(p))
                except SystemExit:
                    pass  # expected: fail() exits with code 1
                else:
                    self.fail("expected SystemExit when fp matches multiple table entries")

    def test_no_verdict_fails(self):
        """Verdict kind is 'no' (architecture not recognised) → SystemExit."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mystery.gguf"
            _write_gguf(p)

            stub = self._make_stub(
                fp_dict={"arch": "unknown_arch", "tensors": 0, "bytes": 0,
                         "expert_bytes": {}, "experts_bpw": None, "bpw": None, "roles": {}},
                kind="no",
            )
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                try:
                    setup.resolve_custom_gguf(str(p))
                except SystemExit:
                    pass  # expected
                else:
                    self.fail("expected SystemExit for unrecognized GGUF")

    def test_missing_file_fails(self):
        """A .gguf path that does not exist → SystemExit with 'does not exist'."""
        try:
            setup.resolve_custom_gguf(str(Path("/nonexistent_dir_nope/nope.gguf")))
        except SystemExit:
            pass  # expected
        else:
            self.fail("expected SystemExit for missing file")

    def test_folder_mode_groups_by_stem(self):
        """A folder with two shard files of the same model → resolved to those shards.

        Shard filenames include '-NNN-of-MMM.gguf'; group-by-stem should collect them together.
        """
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            s1 = d / "model-Q4_K_M-00001-of-00010.gguf"
            s2 = d / "model-Q4_K_M-00002-of-00010.gguf"
            _write_gguf(s1)
            _write_gguf(s2)

            fp_table = json.loads(
                (Path(_ROOT) / "data" / "gguf_fingerprints.json").read_text(encoding="utf-8")
            )
            known = {k: v for k, v in fp_table.items() if not k.startswith("_")}
            first_key, first_entry = next(iter(known.items()))

            stub = self._make_stub(fp_dict=first_entry, kind="known")
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                family, model, shards = setup.resolve_custom_gguf(str(d))

        fam_model = first_key.split("/")
        self.assertEqual(family, fam_model[0])
        self.assertEqual(model, fam_model[1])
        self.assertEqual(len(shards), 2)
        # Shards should be sorted by size (both same size here so order is alphabetical)
        names = [s.name for s in shards]
        self.assertIn("model-Q4_K_M-00001-of-00010.gguf", names)
        self.assertIn("model-Q4_K_M-00002-of-00010.gguf", names)

    def test_folder_no_gguf_files_fails(self):
        """A folder with only non-GGUF files → SystemExit saying 'has no .gguf files'."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "readme.txt").write_text("hello")
            try:
                setup.resolve_custom_gguf(str(d))
            except SystemExit:
                pass  # expected
            else:
                self.fail("expected SystemExit for folder with no .gguf files")

    def test_folder_multiple_models_fails(self):
        """A folder with shards from two different models (different stems) → SystemExit."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            a1 = d / "model_a-00001-of-00010.gguf"
            b1 = d / "model_b-00001-of-00005.gguf"
            _write_gguf(a1)
            _write_gguf(b1)
            try:
                setup.resolve_custom_gguf(str(d))
            except SystemExit:
                pass  # expected
            else:
                self.fail("expected SystemExit for folder with multiple models")

    def test_family_kind_quant_mapping(self):
        """Verdict kind 'family' → derive size from GGUF_QUANT match or filename token."""
        with tempfile.TemporaryDirectory() as tmp:
            # Find a family that is not in the fingerprints table so verdict returns "family".
            # Use qwen36 from FAMILIES; file name must end in one of its model sizes.
            p = Path(tmp) / "model-UD-Q3_K_M.gguf"
            _write_gguf(p)

            fp_dict = {"arch": "some_preview_arch", "tensors": 0, "bytes": 0,
                       "expert_bytes": {}, "experts_bpw": None, "bpw": None, "roles": {}}
            stub = self._make_stub(fp_dict=fp_dict, kind="family")

            # Patch PREVIEW_ARCHS to map our arch to a family title in FAMILIES
            stub.PREVIEW_ARCHS = {"some_preview_arch": setup.FAMILIES["qwen36"]["title"]}

            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                family, model, shards = setup.resolve_custom_gguf(str(p))

        # FAMILIES["qwen36"] has title "Qwen 3.6" (verify) and models dict with sizes
        self.assertTrue(family in setup.FAMILIES)
        self.assertIn(model, setup.FAMILIES[family]["models"])
        self.assertEqual(shards, [p])

    def test_folder_mode_nonquant_size(self):
        """Folder mode: shard filename ending in a non-quant size (e.g. COMPACT) → resolved correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            s1 = d / "Ornith-APEX-I-COMPACT-00001-of-00004.gguf"
            s2 = d / "Ornith-APEX-I-COMPACT-00002-of-00004.gguf"
            _write_gguf(s1)
            _write_gguf(s2)

            fp_dict = {"arch": "ornith_arch", "tensors": 0, "bytes": 0,
                       "expert_bytes": {}, "experts_bpw": None, "bpw": None, "roles": {}}
            stub = self._make_stub(fp_dict=fp_dict, kind="family")
            stub.PREVIEW_ARCHS = {"ornith_arch": setup.FAMILIES["ornith"]["title"]}

            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                family, model, shards = setup.resolve_custom_gguf(str(d))

        self.assertEqual(family, "ornith")
        self.assertEqual(model, "APEX-I-COMPACT")
        self.assertEqual(len(shards), 2)

    def test_conflicting_flags_fail(self):
        """main() with --custom-gguf + conflicting --family/--model → SystemExit (fail)."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.gguf"
            _write_gguf(p)

            fp_table = json.loads(
                (Path(_ROOT) / "data" / "gguf_fingerprints.json").read_text(encoding="utf-8")
            )
            known = {k: v for k, v in fp_table.items() if not k.startswith("_")}
            first_key, first_entry = next(iter(known.items()))

            stub = self._make_stub(fp_dict=first_entry, kind="known")
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                # --family swift conflicts with resolved "qwen" → fail before GPU detection
                argv = ["setup.py", "--custom-gguf", str(p), "--family", "swift", "--model", "IQ3_S"]
                stack.enter_context(mock.patch.object(sys, "argv", argv))
                try:
                    setup.main()
                except SystemExit as e:
                    self.assertEqual(e.code, 1)
                else:
                    self.fail("expected SystemExit on conflicting flags")

    def test_resolved_pair_not_in_models_fails(self):
        """main() with --custom-gguf resolving to a model not in the target family's size list → fail."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.gguf"
            _write_gguf(p)

            # Fake table: key says swift/FAKE_SIZE (not a real swift model)
            fake_fp = {"arch": "swift_arch", "tensors": 42, "bytes": 999,
                       "expert_bytes": {}, "experts_bpw": None, "bpw": None, "roles": {}}
            table = {"swift/FAKE_SIZE": fake_fp}

            stub = self._make_stub(fp_dict=fake_fp, kind="known", table=table)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                # No --family/--model flags → resolved (swift, FAKE_SIZE) must pass validation.
                # FAKE_SIZE is NOT in family_models("swift") → fail before GPU detection.
                argv = ["setup.py", "--custom-gguf", str(p)]
                stack.enter_context(mock.patch.object(sys, "argv", argv))
                try:
                    setup.main()
                except SystemExit as e:
                    self.assertEqual(e.code, 1)
                else:
                    self.fail("expected SystemExit when resolved model not in family's size list")


class InklingCustomGgufTests(unittest.TestCase):
    """Inkling 2 Mini files of a quantization setup does not offer, and Unsloth's top-level layout.  The real
    strata_inspect decides (its verdict and tables); only the header reading is replaced."""

    # the header of Inkling-2-Mini-IQ4_NL.gguf (17,627,940,768 bytes): its routed experts' bytes per type
    IQ4_NL = {"IQ4_NL": 11466178560, "Q8_0": 4331667456}

    def _resolve(self, name: str, arch: str, expert_bytes: dict, check: bool = False):
        tools = str(Path(_ROOT) / "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        import strata_inspect as real
        stub = types.ModuleType("strata_inspect")
        for k in ("verdict", "load_fingerprints", "PREVIEW_ARCHS", "POLY_ARCHS"):
            setattr(stub, k, getattr(real, k))
        stub.read_headers = lambda sources: (arch, [])
        stub.fingerprint = lambda a, t: {"arch": a, "tensors": 704, "bytes": sum(expert_bytes.values()),
                                         "expert_bytes": expert_bytes}
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            p = Path(tmp) / name
            _write_gguf(p)
            stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
            stack.enter_context(mock.patch.dict(setup.FAMILIES["inkling"]["models"]))   # put back after the test
            family, model, shards = setup.resolve_custom_gguf(str(p), check)
            return family, model, dict(setup.FAMILIES["inkling"]["models"]), p.stat().st_size

    def test_iq4_nl_is_a_size_of_its_own(self):
        """An inkling2_loop file of IQ4_NL: installed as Inkling 2 Mini IQ4_NL, not refused nor taken for another size."""
        family, model, sizes, size = self._resolve("Inkling-2-Mini-IQ4_NL.gguf", "inkling2_loop", self.IQ4_NL)
        self.assertEqual((family, model), ("inkling", "IQ4_NL"))
        d = sizes["IQ4_NL"]
        self.assertAlmostEqual(d["arena_gb"], 15.797846016)          # its experts, as read from the header
        self.assertEqual(d["download_gb"], size / 1e9)
        self.assertEqual(d["vram_beside_gb"], 3.7)                    # UD-Q4_K_XL's, the size with the same arena
        self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)))
        self.assertNotIn("IQ4_NL", setup.FAMILIES["inkling"]["models"])   # the table itself is not changed for good

    def test_offered_size_keeps_its_name(self):
        """A file named as an offered size is that size, with no entry added."""
        family, model, sizes, _ = self._resolve("Inkling-2-Mini-UD-Q4_K_XL.gguf", "inkling2_loop", self.IQ4_NL)
        self.assertEqual((family, model), ("inkling", "UD-Q4_K_XL"))
        self.assertEqual(sorted(sizes), ["UD-Q2_K_XL", "UD-Q3_K_XL", "UD-Q4_K_XL"])

    def test_a_name_inside_an_offered_one_is_its_own_size(self):
        """Q2_K is not UD-Q2_K_XL (its letters are inside that name): a size of its own, from its header."""
        family, model, sizes, _ = self._resolve("Inkling-2-Mini-Q2_K.gguf", "inkling2_loop", {"Q2_K": 7000000000})
        self.assertEqual((family, model), ("inkling", "Q2_K"))
        self.assertAlmostEqual(sizes["Q2_K"]["arena_gb"], 7.0)

    def test_loop_rewrite_keeps_its_quant(self):
        """A file rewritten to inkling2_loop and named ...UD-Q2_K_XL.loop.gguf is UD-Q2_K_XL."""
        family, model, sizes, _ = self._resolve("Inkling-2-Mini-UD-Q2_K_XL.loop.gguf", "inkling2_loop",
                                                {"IQ2_S": 2394095616})
        self.assertEqual((family, model), ("inkling", "UD-Q2_K_XL"))
        family, model, sizes, _ = self._resolve("Inkling-2-Mini-UD-IQ3_XXS.loop.gguf", "inkling2_loop", self.IQ4_NL)
        self.assertEqual(model, "UD-IQ3_XXS")

    def test_inspect_points_to_custom_gguf(self):
        """--inspect's advice for an Inkling file is --custom-gguf (--gguf-dir takes only the offered sizes' names)."""
        tools = str(Path(_ROOT) / "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        import strata_inspect
        kind, sentence = strata_inspect.verdict({"arch": "inkling2_loop"}, {})
        self.assertEqual(kind, "family")
        self.assertIn("--custom-gguf", sentence)

    def test_the_warning_says_what_it_read(self):
        """The warning names the quant only when the name has one, says "would be installed" under --check, and says
        which estimate is borrowed from an offered size."""
        said = []
        with mock.patch.object(setup, "warn", said.append):
            _, model, _, _ = self._resolve("Inkling-2-Mini.gguf", "inkling2_loop", self.IQ4_NL)
            self._resolve("Inkling-2-Mini-IQ4_NL.gguf", "inkling2_loop", self.IQ4_NL, check=True)
        self.assertEqual(model, "custom")
        self.assertIn("the file name has no quant", said[0])
        self.assertNotIn("'Mini'", said[0])
        self.assertIn("installed as its own size, custom", said[0])
        self.assertIn("the VRAM before any expert is UD-Q4_K_XL's", said[0])
        self.assertIn("quant 'IQ4_NL'", said[1])
        self.assertIn("would be installed as its own size, IQ4_NL", said[1])

    def test_no_experts_is_refused(self):
        """A header with no routed experts: a clear refusal, not a division by zero later."""
        def fail(msg, hint=""):
            raise SystemExit(msg)

        with mock.patch.object(setup, "fail", fail), self.assertRaises(SystemExit) as cm:
            self._resolve("Inkling-2-Mini-IQ4_NL.gguf", "inkling2_loop", {})
        self.assertIn("no routed experts", str(cm.exception))

    def test_top_level_layout_is_taken(self):
        """Unsloth's top-level files (architecture inkling2), which the engine reads too: an offered size by its
        name, another quantization as a size of its own."""
        family, model, sizes, _ = self._resolve("Inkling-2-Mini-UD-Q2_K_XL.gguf", "inkling2", {"IQ2_S": 2394095616})
        self.assertEqual((family, model), ("inkling", "UD-Q2_K_XL"))
        family, model, sizes, _ = self._resolve("Inkling-2-Mini-IQ4_NL.gguf", "inkling2", self.IQ4_NL)
        self.assertEqual((family, model), ("inkling", "IQ4_NL"))
        self.assertAlmostEqual(sizes["IQ4_NL"]["arena_gb"], 15.797846016)

    def test_every_accepted_architecture_maps_to_inkling(self):
        """Each architecture setup takes for Inkling (POLY_GGUF_ARCHS) is Inkling 2 Mini to --inspect."""
        tools = str(Path(_ROOT) / "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        import strata_inspect
        for arch in setup.POLY_GGUF_ARCHS["inkling"]:
            self.assertEqual(strata_inspect.POLY_ARCHS.get(arch), setup.FAMILIES["inkling"]["title"], arch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
