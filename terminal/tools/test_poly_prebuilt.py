"""Tests for setup.py's ready-made strata-poly (get_poly_prebuilt) from a local release folder: the asset of this
source is installed, one for another source or another card is not, and an engine already there is kept.  No GPU,
no network, no pip.

    python -m unittest tools.test_poly_prebuilt
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

SRC = "0123456789abcdef"
GPU = {"arch": 89, "driver": "581.57"}


def release(folder: Path, src=SRC, archs=(75, 80, 86, 89, 120)) -> None:
    with zipfile.ZipFile(folder / setup.POLY_ASSET.format(src=src), "w") as f:
        f.writestr(setup.POLY_EXE, "new engine")
        f.writestr(setup.VEXE, "new encoder")
        f.writestr("BUILD.json", json.dumps({"source": "release", "archs": list(archs), "cuda": "13.0", "src": src,
                                             "vision_src": src}))


class PolyPrebuilt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.rel = Path(self.tmp.name) / "release"
        self.root.mkdir()
        self.rel.mkdir()
        self.eng = self.root / setup.ENGINE_POLY_DIR
        for p in (mock.patch.object(setup, "ROOT", self.root), mock.patch.object(setup, "WIN", True),
                  mock.patch.object(setup, "POLY_PREBUILT_URL", str(self.rel)),
                  mock.patch.object(setup, "source_hash", lambda parts: SRC),
                  mock.patch.object(setup, "cpu_info", lambda: ("an AVX2 CPU", True, False)),
                  mock.patch.object(setup, "pip_cuda_libs", lambda toolkit=13: None)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_installs_the_asset_of_this_source(self):
        release(self.rel)
        self.assertTrue(setup.get_poly_prebuilt(GPU))
        self.assertEqual((self.eng / setup.POLY_EXE).read_text(), "new engine")
        self.assertEqual((self.eng / setup.VEXE).read_text(), "new encoder")
        self.assertEqual(json.loads((self.eng / "BUILD.json").read_text())["source"], "release")
        self.assertEqual(sorted(p.name for p in self.eng.iterdir()), sorted(["BUILD.json", setup.POLY_EXE, setup.VEXE]))

    def test_none_for_another_source(self):
        release(self.rel, src="fedcba9876543210")
        self.assertFalse(setup.get_poly_prebuilt(GPU))
        self.assertFalse((self.eng / setup.POLY_EXE).exists())

    def test_none_for_a_card_it_has_no_code_for(self):
        release(self.rel)
        self.assertFalse(setup.get_poly_prebuilt({"arch": 90, "driver": "581.57"}))
        self.assertFalse((self.eng / setup.POLY_EXE).exists())

    def test_none_with_a_driver_older_than_cuda_13(self):
        release(self.rel)
        self.assertFalse(setup.get_poly_prebuilt({"arch": 89, "driver": "576.88"}))

    def test_none_for_a_cpu_without_avx2(self):
        release(self.rel)
        with mock.patch.object(setup, "cpu_info", lambda: ("an older CPU", False, False)):
            self.assertFalse(setup.get_poly_prebuilt(GPU))

    def test_a_failed_pip_leaves_the_engine_there(self):
        self.eng.mkdir()
        (self.eng / setup.POLY_EXE).write_text("compiled here")
        release(self.rel)
        with mock.patch.object(setup, "pip_cuda_libs", side_effect=SystemExit(1)):
            with self.assertRaises(SystemExit):
                setup.get_poly_prebuilt(GPU)
        self.assertEqual((self.eng / setup.POLY_EXE).read_text(), "compiled here")

    def test_an_update_downloads_in_place_of_a_ready_made_engine_alone(self):
        release(self.rel)
        self.eng.mkdir()
        cfg = {"exe": str(self.eng / setup.POLY_EXE), "args": []}
        for source, want in (("local", "compiled here"), ("release", "new engine")):
            (self.eng / setup.POLY_EXE).write_text("compiled here")
            (self.eng / "BUILD.json").write_text(json.dumps({"source": source, "archs": [89], "src": "older"}))
            with mock.patch.object(setup, "gpu_info", lambda i=None: GPU), \
                    mock.patch.object(setup, "build_engine_poly", lambda *a: self.eng), \
                    mock.patch.object(setup, "get_llama_cpp", lambda: None):
                setup.update_poly_engine(dict(cfg))
            self.assertEqual((self.eng / setup.POLY_EXE).read_text(), want, source)

    def test_an_update_without_a_release_of_its_source_keeps_the_ready_made_engine(self):
        release(self.rel, src="fedcba9876543210")
        self.eng.mkdir()
        (self.eng / setup.POLY_EXE).write_text("ready-made")
        (self.eng / "BUILD.json").write_text(json.dumps({"source": "release", "archs": [89], "src": "older"}))
        build = mock.Mock()
        with mock.patch.object(setup, "gpu_info", lambda i=None: GPU), \
                mock.patch.object(setup, "build_engine_poly", build):
            setup.update_poly_engine({"exe": str(self.eng / setup.POLY_EXE), "args": []})
        build.assert_not_called()
        self.assertEqual((self.eng / setup.POLY_EXE).read_text(), "ready-made")

    def test_a_ready_made_engine_with_an_older_encoder_is_replaced(self):
        release(self.rel)
        self.eng.mkdir()
        (self.eng / setup.POLY_EXE).write_text("ready-made")
        (self.eng / "BUILD.json").write_text(json.dumps({"source": "release", "archs": [89], "src": SRC,
                                                         "vision_src": "older"}))
        self.assertTrue(setup.get_poly_prebuilt(GPU))
        self.assertEqual((self.eng / setup.VEXE).read_text(), "new encoder")

    def test_an_engine_of_this_source_is_kept(self):
        self.eng.mkdir()
        (self.eng / setup.POLY_EXE).write_text("compiled here")
        (self.eng / "BUILD.json").write_text(json.dumps({"source": "local", "archs": [89], "src": SRC}))
        release(self.rel)
        self.assertTrue(setup.get_poly_prebuilt(GPU))
        self.assertEqual((self.eng / setup.POLY_EXE).read_text(), "compiled here")

    def test_a_compile_after_it_is_for_this_card_alone(self):
        release(self.rel)
        setup.get_poly_prebuilt(GPU)
        meta = json.loads((self.eng / "BUILD.json").read_text())
        meta["src"] = "a source it was not built from"
        (self.eng / "BUILD.json").write_text(json.dumps(meta))
        with mock.patch.object(setup, "install_build_tools", side_effect=RuntimeError("compiles")) as tools:
            with self.assertRaises(RuntimeError):
                setup.build_engine_poly(GPU, True, None)
        self.assertEqual(tools.call_args[0][0]["archs"], [89])

    def compile_here(self, gpu, toolkit, vision="none"):
        """build_engine_poly() with a compile that writes what it was asked: (the CMake definitions, the toolkit)."""
        made = {}

        def cmake(src, bdir, target, defs, vcvars, script):
            made.setdefault(target, defs)
            out = bdir / "bin" if target == "strata-vision" else bdir
            out.mkdir(parents=True, exist_ok=True)
            (out / (setup.VEXE if target == "strata-vision" else setup.POLY_EXE)).write_text(f"{target} {toolkit}")
        with mock.patch.object(setup, "install_build_tools", lambda g, yes: (str(self.root / "nvcc"), None)), \
                mock.patch.object(setup, "cmake_build", cmake), mock.patch.object(setup, "cuda_lib_dirs", lambda t: []), \
                mock.patch.object(setup, "source_version", lambda: "0.1.42"):
            setup.build_engine_poly(gpu, True, None, vision, toolkit)
        return made

    def test_cuda_12_compiles_in_place_of_a_ready_made_engine(self):
        release(self.rel)
        setup.get_poly_prebuilt(GPU)
        made = self.compile_here(GPU, 12, "gpu")
        self.assertEqual(sorted(made), ["strata-poly", "strata-vision"])
        self.assertNotIn("-DSTRATA_EXPERIMENTAL_SM60=ON", made["strata-poly"])
        meta = json.loads((self.eng / "BUILD.json").read_text())
        self.assertEqual((meta["toolkit"], meta["archs"], meta["source"]), (12, [89], "local"))
        # the ready-made one is not taken back while the model asks for CUDA 12, and is when it no longer does
        self.assertEqual(self.compile_here(GPU, 12, "gpu"), {})
        self.assertTrue(setup.get_poly_prebuilt(GPU))
        self.assertEqual((self.eng / setup.POLY_EXE).read_text(), "new engine")

    def test_cuda_12_for_a_pascal_card_is_the_experimental_build(self):
        made = self.compile_here({"arch": 61, "driver": "550.54"}, 12)
        self.assertIn("-DSTRATA_EXPERIMENTAL_SM60=ON", made["strata-poly"])
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=61", made["strata-poly"])


class OnLinux(unittest.TestCase):
    """#181: on x86-64 Linux the release's tarball of this source is installed the same way, when its programs start
    here (glibc, libstdc++ and pip's CUDA 13 libraries); otherwise setup compiles."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repo"
        self.rel = Path(self.tmp.name) / "release"
        self.root.mkdir()
        self.rel.mkdir()
        self.eng = self.root / setup.ENGINE_POLY_DIR
        asset = "strata-poly-linux-x64-cuda13-{src}.tar.gz"
        with tarfile.open(self.rel / asset.format(src=SRC), "w:gz") as t:
            for name, data in ((setup.POLY_EXE, b"new engine"), (setup.VEXE, b"new encoder"),
                               ("BUILD.json", json.dumps({"source": "release", "archs": [75, 80, 86, 89, 120], "cuda": "13.0",
                                                          "src": SRC, "vision_src": SRC, "platform": "linux-x64"}).encode())):
                info = tarfile.TarInfo("./" + name)
                info.size, info.mode = len(data), 0o755 if name != "BUILD.json" else 0o644
                t.addfile(info, io.BytesIO(data))
        for p in (mock.patch.object(setup, "ROOT", self.root), mock.patch.object(setup, "WIN", False),
                  mock.patch("sys.platform", "linux"), mock.patch.object(setup, "is_arm64", lambda: False),
                  mock.patch.object(setup, "POLY_ASSET", asset), mock.patch.object(setup, "POLY_PREBUILT_URL", str(self.rel)),
                  mock.patch.object(setup, "source_hash", lambda parts: SRC),
                  mock.patch.object(setup, "cpu_info", lambda: ("an AVX2 CPU", True, False)),
                  mock.patch.object(setup, "pip_cuda_libs", lambda toolkit=13: None),
                  mock.patch.object(setup, "poly_will_not_start", lambda folder: "")):
            p.start()
            self.addCleanup(p.stop)

    def test_installs_the_tarball_of_this_source(self):
        self.assertTrue(setup.get_poly_prebuilt(GPU))
        self.assertEqual((self.eng / setup.POLY_EXE).read_text(), "new engine")
        self.assertEqual((self.eng / setup.VEXE).read_text(), "new encoder")
        self.assertEqual(sorted(p.name for p in self.eng.iterdir()), sorted(["BUILD.json", setup.POLY_EXE, setup.VEXE]))
        if os.name == "posix":
            self.assertTrue(os.access(self.eng / setup.POLY_EXE, os.X_OK), "the engine keeps its execute bit")

    def test_none_when_its_programs_do_not_start_here(self):
        with mock.patch.object(setup, "poly_will_not_start", lambda folder: "version `GLIBC_2.34' not found"):
            self.assertFalse(setup.get_poly_prebuilt(GPU))
        self.assertFalse(self.eng.exists() and any(self.eng.iterdir()), "nothing is left for the compile to trip on")

    def test_none_on_arm64(self):
        with mock.patch.object(setup, "is_arm64", lambda: True):
            self.assertFalse(setup.get_poly_prebuilt(GPU))


@unittest.skipUnless(os.name == "posix", "the Linux loader check runs the programs")
class WillNotStart(unittest.TestCase):
    def test_what_the_loader_said_or_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            for name, code in ((setup.POLY_EXE, 2), (setup.VEXE, 0)):    # strata-poly --help exits 2
                (folder / name).write_text(f"#!/bin/sh\nexit {code}\n")
                (folder / name).chmod(0o755)
            self.assertEqual(setup.poly_will_not_start(folder), "")
            (folder / setup.VEXE).write_text("#!/bin/sh\necho 'strata-vision: error while loading shared libraries: "
                                             "libcublas.so.13: cannot open shared object file' >&2\nexit 127\n")
            self.assertIn("libcublas.so.13", setup.poly_will_not_start(folder))


class FromARelease(unittest.TestCase):
    """The ready-made strata-poly from a GitHub release, as a released copy of PolyStrata gets it: the zip is checked
    against the size and SHA-256 the releases API publishes for that tag before it is opened."""

    URL = "https://github.com/simplexai-labs/PolyStrata/releases/download/v0.2.17-poly.1/"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repo"
        self.rel = Path(self.tmp.name) / "release"
        self.root.mkdir()
        self.rel.mkdir()
        release(self.rel)
        self.zip = self.rel / setup.POLY_ASSET.format(src=SRC)
        self.eng = self.root / setup.ENGINE_POLY_DIR
        self.asked = []
        for p in (mock.patch.object(setup, "ROOT", self.root), mock.patch.object(setup, "WIN", True),
                  mock.patch.object(setup, "POLY_PREBUILT_URL", self.URL),
                  mock.patch.object(setup, "source_hash", lambda parts: SRC),
                  mock.patch.object(setup, "cpu_info", lambda: ("an AVX2 CPU", True, False)),
                  mock.patch.object(setup, "pip_cuda_libs", lambda toolkit=13: None),
                  mock.patch("urllib.request.urlopen", lambda req, timeout=None: mock.MagicMock()),
                  mock.patch.object(setup, "download", self.download)):
            p.start()
            self.addCleanup(p.stop)

    def download(self, url, dst, what=None):
        self.assertEqual(url, self.URL + self.zip.name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(self.zip.read_bytes())

    def api(self, digest):
        """setup._urlopen answering the releases API with this digest for the asset (None: the API cannot be reached)."""
        def answer(req, timeout=None):
            self.asked.append(req.full_url)
            if digest is None:
                raise OSError("offline")
            body = {"assets": [{"name": self.zip.name, "size": self.zip.stat().st_size, "digest": digest}]}
            r = mock.MagicMock()
            r.__enter__.return_value.read.return_value = json.dumps(body).encode()
            return r
        return mock.patch.object(setup, "_urlopen", answer)

    def test_the_default_is_this_repository_s_latest_release(self):
        env = {k: v for k, v in os.environ.items() if k != "STRATA_POLY_PREBUILT_URL"}
        url = subprocess.run([sys.executable, "-c", "import setup; print(setup.POLY_PREBUILT_URL)"], cwd=ROOT, env=env,
                             capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(setup.release_of(url),
                         ("https://api.github.com/repos/simplexai-labs/PolyStrata/releases", None))

    def test_the_published_sha256_is_checked_for_that_tag(self):
        sha = hashlib.sha256(self.zip.read_bytes()).hexdigest()
        out = io.StringIO()
        with self.api("sha256:" + sha), contextlib.redirect_stdout(out):
            self.assertTrue(setup.get_poly_prebuilt(GPU))
        self.assertEqual(self.asked, ["https://api.github.com/repos/simplexai-labs/PolyStrata/releases/tags/v0.2.17-poly.1"])
        self.assertIn("SHA-256 checksum verified against GitHub's", out.getvalue())
        self.assertEqual((self.eng / setup.POLY_EXE).read_text(), "new engine")

    def test_a_zip_that_is_not_the_published_one_is_not_opened(self):
        out = io.StringIO()
        with self.api("sha256:" + "0" * 64), contextlib.redirect_stdout(out):
            self.assertFalse(setup.get_poly_prebuilt(GPU))
        self.assertIn("wrong SHA-256", out.getvalue())
        self.assertFalse((self.eng / setup.POLY_EXE).exists())
        self.assertFalse((self.eng / "_unpack").exists())
        self.assertFalse((self.eng / self.zip.name).exists(), "the refused zip is deleted")

    def test_no_answer_from_the_api_says_so_and_installs(self):
        out = io.StringIO()
        with self.api(None), contextlib.redirect_stdout(out):
            self.assertTrue(setup.get_poly_prebuilt(GPU))
        self.assertIn("NOT verified", out.getvalue())


if __name__ == "__main__":
    unittest.main()
