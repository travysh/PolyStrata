"""setup.py on a Mac: Apple Silicon, the Metal engine (metal/), built from source on the Mac.

    ./setup.sh [options]          (setup.py hands a macOS run to this file; the PC paths are not used)

What it does, each step skipped when it is already done:

  1. checks the Mac: Apple Silicon, the chip, unified memory, macOS version, the Command Line Tools, free disk
  2. recommends a model by memory (recommend, never force: an explicit --model is kept, with what it risks)
  3. installs the Python packages into .venv (cmake and ninja come from pip, just before step 4: no Homebrew, no sudo)
  4. builds the Metal engine from source: metal-cpp (Apple's headers) and the vendored ggml (third_party/strata-ggml)
  5. downloads the model (setup.py's downloader: resumable, the pinned revision)
  6. prepares the model: the native pack (tools/iq_pack.py) and the MTP draft layer (tools/mtp_fetch.py, mtp_pack.py)
  7. writes strata-<model>.json and run-<model>.sh with the Metal defaults (--spec 2, the Q8_0 hyper-connection
     read) and starts the server (serve/server.py), the same as on a PC

Smaller families selected with --family use the native strata-poly Metal engine and GGUF files directly.
The existing Qwen3.8 path and defaults stay unchanged.

Everything shared - the model table, the downloader, the config and run-script writers, the server - is setup.py's.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # engine/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "terminal"))        # setup.py, strata_control and the tools are the Terminal's
import setup as S  # noqa: E402
from strata_control import mac_fit as _fit  # noqa: E402
from strata_control.mac_fit import (  # noqa: E402,F401  (re-exported: the tests and callers use these names here)
    BIG_MAC_GB, CONTEXTS, GIB_GB, GPU_SHARE, MAC_ABOUT, MAC_MODELS, MIN_MACOS, OVERHEAD_GB, RECOMMENDED,
    RESIDENT_LEAVE_SHARE, STREAM_COPY_GAIN, STREAM_MEASURED, STREAM_MTP_FROM_GB, STREAM_SPEC, TESTED_CONTEXT,
    _sysctl, gpu_budget_gb, mac_info, mac_problems, macos_tuple, recommend_context, recommend_model, stream_mtp)
from tools import metal_poly_setup as P  # noqa: E402

# Apple's metal-cpp headers for the macOS 26 SDK (https://developer.apple.com/metal/cpp/), pinned by SHA-256
METAL_CPP_URL = "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip"
METAL_CPP_SHA256 = "4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4"
BUILD_DIR = ROOT / "build-metal"
EXE = BUILD_DIR / "strata"

# the Unsloth files at setup.py's pinned revision: name -> (bytes, SHA-256), the Hub's LFS pointers (UD-Q4_K_XL's
# are setup.py's UNSLOTH_SHARDS)
UD_SHARDS = {
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf": (10946618, "610922cdb1afe1094e95d7a3e86b12ee33317ebbf7320ac9344e21b6ae4bb69f"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00002-of-00006.gguf": (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00003-of-00006.gguf": (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00004-of-00006.gguf": (49990245824, "9ee6bbe462e6830864382d629101490625b1c95afa15492b7ad550c7bed2dd13"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00005-of-00006.gguf": (49882416192, "0b2437f661f584030cb5aeb8fb1bb70f6422518a6746bfcc0304167c7b6b8beb"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00006-of-00006.gguf": (3320101792, "55dd96c5ad0863b06b6393151b9ad0ac727b8d58a62e02a296ca5d5ddbfee4a3"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf": (10946624, "8bdc6bad55b6699f46f60708829932ff4018d7ac9bd309c388534fa31753bdb7"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf": (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf": (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf": (49814108512, "0ea4b599880a5a52fcd9c188ba1443f646c69c96571904ddbd6a79246a01a997"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf": (49419317344, "9948e81ae8368144b135a7403d8c0f2cf9a6c2f9f692313865855b2832e696c9"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf": (14838313984, "e4ae255234f42b18012f6d6eec5bf615b57479b7e0a0233c47436a321ffeec3f"),
}

# Smaller architectures use strata-poly, not the fixed Qwen3.8 Metal window.
# Keep their files, revisions, sizes and checksums in setup.py's single catalog.
# The native port is opt-in through --family; the existing Qwen3.8 defaults stay unchanged.
MAC_POLY_FAMILIES = tuple(f for f, d in S.FAMILIES.items() if d.get("engine") == "poly")


def smaller_model_catalog() -> dict:
    """The small strata-poly families available in the shared installer."""
    catalog = {}
    for family in MAC_POLY_FAMILIES:
        fam = S.FAMILIES[family]
        catalog[family] = {
            "title": fam["title"],
            "engine": "strata-poly",
            "metal_ready": True,
            "model_validated": False,
            "models": {
                model: {"download_gb": spec["download_gb"],
                        "files": [S.model_file(fam, model, i)
                                  for i in range(1, S.model_shards(fam, model) + 1)]}
                for model, spec in S.family_models(family).items()
            },
        }
    return catalog


# Size estimates are distinct from catalog identity. Published files and hashes always
# come from setup.py; the measured values of mac_fit's table keep the established defaults.
for _family in ("qwen", "swift", "coder", "unsloth", "nvfp4"):
    if _family not in S.FAMILIES:
        continue
    for _model, _spec in S.family_models(_family).items():
        MAC_MODELS.setdefault(_model, {"shard1_gb": _spec["arena_gb"] + 5.6, "tested": False, "family": _family,
                                      "estimate": True})   # (mac_fit.app_sizes leaves these out of the app's list)


def model_catalog() -> dict:
    """All installer choices, keyed by family and then quantization (never quant alone)."""
    result = {}
    for family, fam in {**S.FAMILIES, **S.OWN_PATH_FAMILIES}.items():
        engine = "strata-q36" if family in S.OWN_PATH_FAMILIES else (
            "strata-poly" if fam.get("engine") == "poly" else "strata")
        models = fam["models"] if family in S.OWN_PATH_FAMILIES else S.family_models(family)
        result[family] = dict(fam, models=models, metal_engine=engine)
    return result


def model_shard_names(model: str, family: str | None = None) -> list[str]:
    """The chosen family's files in order (shard 1 first); ready packs use their manifest instead."""
    family = family or MAC_MODELS[model].get("family", "qwen")
    fam = S.FAMILIES[family]
    if family == "gyro":                                   # agentionai's one file (the n-gram table is inside it)
        return [S.model_file(fam, model, 1)]
    if fam.get("feed"):                                    # a model of the signed feed: its files as the feed names them
        return [S.model_file(fam, model, i) for i in range(1, MAC_MODELS[model].get("shards", 2) + 1)]
    return [S.model_file(fam, model, i) for i in range(1, S.model_shards(fam, model) + 1)]


def shard_sha(name: str):
    return UD_SHARDS.get(name) or getattr(S, "UNSLOTH_SHARDS", {}).get(name) or getattr(S, "UNSLOTH_V31_SHARDS", {}).get(name)


# ------------------------------------------------------------------------------------------------ the Mac
def model_metadata(model: str, family: str | None = None) -> dict:
    """The Mac table's entry for this size.  A family that only shares the size's name gets an estimate: equal names do
    not make one family's measurements valid for another."""
    measured = MAC_MODELS[model]
    if family is None or measured.get("family", "qwen") == family:
        return measured
    return {"shard1_gb": S.family_models(family)[model]["arena_gb"] + 5.6, "tested": False}


def model_need_gb(model: str, family: str | None = None) -> float:   # mac_fit's, with the family
    if model_metadata(model, family) is MAC_MODELS[model]:
        return _fit.model_need_gb(model)
    return model_metadata(model, family)["shard1_gb"] + OVERHEAD_GB


def streams(model: str, ram_gb: float, family: str | None = None) -> bool:
    """The pack does not fit Metal's share of the memory: the engine streams its experts from the SSD."""
    if model_metadata(model, family) is MAC_MODELS[model]:
        return _fit.streams(model, ram_gb)
    return model_need_gb(model, family) > gpu_budget_gb(ram_gb)


def model_note(model: str, ram_gb: float, family: str | None = None) -> list[str]:
    """What a chosen pack risks on this Mac (said, never refused)."""
    if model_metadata(model, family) is MAC_MODELS[model]:
        return _fit.model_note(model, ram_gb)
    notes = []
    if streams(model, ram_gb, family):
        notes.append(f"{model} needs ~{model_need_gb(model, family):.0f} GB of the {gpu_budget_gb(ram_gb):.0f} GB Metal keeps "
                     "resident on this Mac: its experts stream from the SSD (automatic); not measured streaming")
    notes.append(f"{model} has not been measured on a Mac yet")
    return notes


# ------------------------------------------------------------------------------------------------ the engine
def metal_cpp() -> Path:
    """Apple's metal-cpp headers in third_party/metal-cpp (one zip, pinned)."""
    dst = ROOT / "third_party" / "metal-cpp"
    if (dst / "Metal" / "Metal.hpp").exists():
        return dst
    z = ROOT / "third_party" / "metal-cpp_26.zip"
    S.download(METAL_CPP_URL, z, "metal-cpp (Apple's Metal headers)")
    S.verify_sha256(z, z.stat().st_size, METAL_CPP_SHA256)
    tmp = ROOT / "third_party" / "_metal_cpp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(tmp / "metal-cpp"), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    return dst


def tool(name: str) -> str:
    """cmake / ninja from the venv's pip packages (no Homebrew)."""
    p = Path(sys.executable).parent / name
    return str(p) if p.exists() else (shutil.which(name) or name)


def engine_current(poly: bool = False, *, engine: str | None = None) -> bool:
    engine = engine or ("strata-poly" if poly else "strata")
    exe = BUILD_DIR / engine
    stamp = "BUILD.json" if engine == "strata" else f"BUILD-{engine.removeprefix('strata-')}.json"
    try:
        meta = json.loads((BUILD_DIR / stamp).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return exe.exists() and meta.get("version") == S.source_version() and meta.get("source") == source_digest()


def source_digest() -> str:
    """What the engine is built from: every source file's path, size and modification time (src/, include/, metal/,
    third_party/ggml, third_party/strata-ggml).  A git pull, an unzipped update or a local edit changes it; nothing else does."""
    h = hashlib.sha256()
    for d in ("src", "include", "metal", "third_party/ggml", "third_party/strata-ggml"):
        for f in sorted((ROOT / d).rglob("*")):
            if f.is_file() and f.suffix in (".cpp", ".cu", ".h", ".hpp", ".mm", ".metal", ".inc", ".txt", ".cuh", ".py"):
                st = f.stat()
                h.update(f"{f.relative_to(ROOT).as_posix()}|{st.st_size}|{int(st.st_mtime)}\n".encode())
    return h.hexdigest()


def build_engine(llama: Path, jobs: int | None = None, *, poly: bool = False, engine: str | None = None) -> Path:
    """The Metal engine from source: cmake + ninja (pip's), Apple's clang from the Command Line Tools."""
    engine = engine or ("strata-poly" if poly else "strata")
    exe = BUILD_DIR / engine
    stamp = "BUILD.json" if engine == "strata" else f"BUILD-{engine.removeprefix('strata-')}.json"
    if engine_current(engine=engine):
        S.ok(f"Metal engine: {exe} (built for this source)")
        return exe
    mcpp = metal_cpp()
    S.ensure_build_tools()                                # cmake and ninja are in requirements-build.txt, not requirements.txt (#3403)
    S.say("  Building the Metal engine from source (5-15 minutes the first time) ...")
    BUILD_DIR.mkdir(exist_ok=True)
    log = BUILD_DIR / "build.log"
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    cfg = [tool("cmake"), "-S", str(ROOT / "metal"), "-B", str(BUILD_DIR), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
           f"-DCMAKE_MAKE_PROGRAM={tool('ninja')}", f"-DMETAL_CPP_DIR={mcpp}", *S.ggml_defs(llama)]
    cfg += [f"-DSTRATA_METAL_POLY={'OFF' if engine == 'strata' else 'ON'}"]
    build = [tool("ninja"), "-C", str(BUILD_DIR), exe.name] + (["-j", str(jobs)] if jobs else [])
    with open(log, "w", encoding="utf-8") as f:
        for cmd in (cfg, build):
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            if rc != 0:
                S.fail(f"the Metal engine did not build (see {log})",
                       "the last lines of the log name the problem; the Command Line Tools must be up to date "
                       "(softwareupdate --list)")
    (BUILD_DIR / stamp).write_text(json.dumps({"version": S.source_version(), "source": source_digest(),
                                                      "backend": "metal"}, indent=1), encoding="utf-8")
    S.ok(f"Metal engine built: {exe}")
    return exe


# ------------------------------------------------------------------------------------------------ the config
def engine_args(pack: Path, shards: list, rt: Path, ctx: int, stream: bool = False, ram_gb: float = 0.0,
                model: str = "") -> list[str]:
    """The Metal defaults (strata_control/mac_fit.py), with this checkout's expert profile."""
    return _fit.engine_args(pack, shards, rt, ctx, stream, ram_gb, ROOT.parent / "terminal" / "data" / "expert-profile.bin", model)


# the image encoder's projector (setup.py's pinned revision): name -> (bytes, SHA-256)
VISION_SHA256 = {"mmproj-Qwen3.8-Flash-Next-BF16.gguf":
                 (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0")}
VISION_EXE = ROOT / "build-vision" / "bin" / "strata-vision"


def build_vision(llama: Path) -> Path:
    """strata-vision (terminal/tools/vision: the Strata project's own vision towers, no llama.cpp or ggml), with the towers' Metal
    kernels: --gpu reads the picture on the GPU, and without it the same program reads on the CPU."""
    stamp = ROOT / "build-vision" / "BUILD.json"
    src = source_digest()
    try:
        if VISION_EXE.exists() and json.loads(stamp.read_text(encoding="utf-8")).get("source") == src:
            S.ok(f"image encoder: {VISION_EXE}")
            return VISION_EXE
    except (OSError, ValueError):
        pass
    S.ensure_build_tools()
    S.say("  Building the image encoder (1-3 minutes) ...")
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    log = ROOT / "build-vision.log"
    with open(log, "w", encoding="utf-8") as f:
        for cmd in ([tool("cmake"), "-S", str(ROOT.parent / "terminal" / "tools" / "vision"), "-B", str(ROOT / "build-vision"), "-G", "Ninja",
                     "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_MAKE_PROGRAM={tool('ninja')}",
                     "-DSTRATA_VISION_CUDA=OFF", "-DSTRATA_VISION_METAL=ON", "-DSTRATA_PORTABLE=OFF"],
                    [tool("ninja"), "-C", str(ROOT / "build-vision"), "strata-vision"]):
            if subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env) != 0:
                S.fail(f"the image encoder did not build (see {log})")
    stamp.write_text(json.dumps({"source": src}), encoding="utf-8")
    S.ok(f"image encoder built: {VISION_EXE}")
    return VISION_EXE


def ple_shard(files: list) -> Path:
    """The shard that holds the n-gram table (per_layer_token_embd)."""
    sys.path.insert(0, str(ROOT / "metal" / "tools"))
    import metal_repack as R
    for f in files:
        rd = R.Reader(f)
        rd.f.close()
        if any(t["name"] == R.PLE for t in rd.tensors):
            return f
    S.fail(f"no shard of {files[0].name} holds the n-gram table (per_layer_token_embd)")


# the experts' streaming copy (metal/tools/stream_experts.py): what stays free on the disk after it is written
EXPERTS_MARGIN_GB = 10.0


def experts_file_fits(need_gb: float, free_gb: float) -> bool:
    return free_gb >= need_gb + EXPERTS_MARGIN_GB


def experts_file(native: Path) -> None:
    """A streaming Mac: <model>.experts beside the GPU's GGUF - every expert's rows in one piece, read with one SSD
    request per expert instead of three (the engine uses it when it is there and matches; else the GGUF).  Written
    only when the disk has room for it and EXPERTS_MARGIN_GB more; otherwise setup says so and goes on."""
    out = native.with_name(native.name + ".experts")
    if out.exists() and S.done(out):
        S.ok(f"the experts' streaming copy: {out.name}")
        return
    sys.path.insert(0, str(ROOT / "metal" / "tools"))
    import stream_experts as X
    need = X.size_bytes(native) / 1e9
    free = S.free_gb(native.parent)
    if not os.access(native.parent, os.W_OK):
        S.warn(f"{native.parent} cannot be written: the experts' streaming copy ({need:.0f} GB) is not written; the "
               f"engine streams from the model file ({STREAM_COPY_GAIN} slower)")
        return
    if not experts_file_fits(need, free):
        S.warn(f"the experts' streaming copy needs {need:.0f} GB and {EXPERTS_MARGIN_GB:.0f} GB to spare on the disk "
               f"({free:.0f} GB free): not written; the engine streams from the model file ({STREAM_COPY_GAIN} "
               f"slower). Free some space and run ./setup.sh --setup again to add it")
        return
    S.say(f"  Writing the experts in SSD-read order (one time, {need:.0f} GB; each expert is read with one request) ...")
    S.run([sys.executable, str(ROOT / "metal" / "tools" / "stream_experts.py"), "--gguf", str(native), "--verify"])
    S.mark(out)
    S.ok(f"the experts' streaming copy: {out.name}")


def config_path(tag: str) -> Path:
    """A model's run config: in setup.home() (terminal/), where setup.py writes the PC's and where the server, polystrata.py
    and the next setup look for them, not beside this file in engine/."""
    return S.home() / f"strata-{tag.lower()}.json"


def mac_config(exe: Path, args: list, pack: Path, model_name: str, tag: str, port: int) -> dict:
    return {"exe": str(exe), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
            "model_name": model_name, "log": str(S.home() / f"strata-{tag.lower()}.log"), "lib_dirs": [], "port": port,
            "backend": "metal"}


def mac_configs() -> list[Path]:
    out = []
    for p in S.installed_configs():
        try:
            if json.loads(p.read_text(encoding="utf-8-sig")).get("backend") == "metal":
                out.append(p)
        except (OSError, ValueError):
            pass
    return out


REPACK_LOCK = ".repack.lock"       # metal_repack.py: <out>.repack.lock holds the repack's pid while it runs
PREPARE_LOCK = ".preparing.lock"   # setup: beside the repacked file, holds setup's pid from the repack to the end of the pack


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True            # it exists, it is somebody else's
    except OSError:
        return False
    return True


def repack_running(folder: Path) -> list[int]:
    """Pids of repacks (metal_repack.py) or setup's own model preparation (repack + pack) still running for the files in
    `folder`.  A stale lock (its process gone) is removed.  The repack and the pack move 100+ GB through the disk and one
    holds up to 20 GB of memory: an engine with a 70-90 GB footprint beside them is what froze the 128 GB M5 Max twice
    (watchdog panics, 2026-10-13)."""
    pids = []
    if not folder.is_dir():
        return pids
    for lock in [*folder.glob("*" + REPACK_LOCK), folder / PREPARE_LOCK]:
        try:
            pid = int(lock.read_text().strip())
        except (OSError, ValueError):
            continue
        if pid != os.getpid() and _pid_alive(pid):
            pids.append(pid)
        elif pid != os.getpid():
            try:
                lock.unlink()
            except OSError:
                pass
    return pids


def engine_pids(run=subprocess.run) -> list[int]:
    """Pids of the running Metal engines (`strata`, the executable setup builds) that belong to THIS user.  On a shared Mac
    another person's engine is theirs to stop: `pgrep -x strata` alone saw it and sent setup away with "stop the server" for a
    window the person could not close (M5 Max, 2026-10-31)."""
    cmd = ["pgrep", "-x", "strata"]
    if hasattr(os, "getuid"):
        cmd[1:1] = ["-u", str(os.getuid())]
    try:
        out = run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(x) for x in out.split() if x.isdigit() and int(x) != os.getpid()]


def start(cfg_path: Path, port: int | None, open_browser: bool = True) -> int:
    """serve/server.py with this config (setup.py's start without the NVIDIA / AMD card checks)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    missing = [p for p in [cfg["exe"], *[a for a in cfg["args"] if a.endswith(".gguf")]] if not Path(p).exists()]
    if missing:
        S.fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run ./setup.sh --setup to repair it")
    busy = [p for g in cfg["args"] if g.endswith(".gguf") for p in repack_running(Path(g).parent)]
    if busy:       # never start an engine while a repack (or the pack after it) is writing: wait would hide a stuck one
        S.fail(f"a model repack is still running here (pid {busy[0]}): the engine is not started beside it",
               "wait for it to finish (or stop it), then start again")
    cfg_path.touch()
    if "--mtp" in cfg["args"][:-1]:
        S.refresh_draft_vocab(Path(cfg["args"][cfg["args"].index("--mtp") + 1]), cfg.get("draft_vocab", "cjk"))
    cmd = [sys.executable, str(ROOT.parent / "terminal" / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port or cfg.get("port", 8080))]
    if open_browser and cfg.get("open_browser") is True:     # only after --browser: the Monitor page
        cmd.append("--open")
    S.say()
    S.say(f"  Starting {cfg.get('model_name', 'the model')} on Metal: the engine maps the model file and is ready in about "
          "half a minute (longer the first time after a restart, while macOS reads the file).")
    S.say("  Closing this window stops the model.")
    return subprocess.call(cmd)


def setup_poly(a, info: dict, data: Path, elsewhere: list) -> int:
    """Shared GGUF catalog and server, with native strata-poly Metal kernels."""
    fam = S.FAMILIES[a.family]
    sizes = S.family_models(a.family)
    if a.model and a.model not in sizes:
        S.fail(f"{fam['title']} has no {a.model} file", "choose one of: " + ", ".join(sizes))
    # Unified memory is shared by the mapped weights, GPU buffers and macOS.
    # Prefer a smaller download; do not apply the discrete-VRAM sizing rule.  A family that names the sizes run on a
    # Mac ("mac_measured", largest first) gets the first of them that fits this Mac's memory.
    measured = fam.get("mac_measured") or ()
    fits = lambda m: sizes[m]["download_gb"] + OVERHEAD_GB <= info["ram_gb"] * GIB_GB   # noqa: E731
    smallest = min(sizes, key=lambda m: sizes[m]["download_gb"])
    model = a.model or next((m for m in measured if m in sizes and fits(m)), smallest)
    if not a.model and not a.yes:
        names = list(sizes)
        for i, m in enumerate(names, 1):
            S.say(f"  {i}) {m}: {sizes[m]['download_gb']:.1f} GB"
                  + (", not measured on a Mac" if measured and m not in measured else ""))
        model = names[int(S.ask("Which size?", [str(i) for i in range(1, len(names)+1)],
                                str(names.index(model)+1), a.yes))-1]
    if measured and model not in measured:
        S.warn(f"{model} has not been measured on a Mac ({' and '.join(measured)} have)")
    d = sizes[model]
    S.warn("strata-poly on Metal is experimental: kernel parity checks pass; broader full-model validation is pending (docs/MAC.md)")
    if d["download_gb"] + OVERHEAD_GB > info["ram_gb"] * GIB_GB:
        S.warn("the model plus working memory may exceed this Mac's unified memory")
    tag = fam["tag"] + model
    ctx = a.context or 8192
    if ctx <= 0:
        S.fail("--context must be positive")
    vision = a.vision == "yes"
    # a family whose picture tower has no Metal kernels ("mac_vision_cpu") reads pictures on the CPU: yes, gpu and cpu
    # all mean that, and with none of them it is asked, as on a PC (no by default)
    cpu_tower = bool(fam.get("vision") and fam.get("mac_vision_cpu"))
    if cpu_tower:
        if a.vision in ("yes", "gpu", "cpu"):
            vision = True
        elif a.vision is None:
            vision = S.ask("Should the model also read pictures? (its image encoder, on the CPU: a few seconds a "
                           "picture)", ["y", "n"], "n", a.yes) == "y"
        if vision and a.vision == "gpu":
            S.warn(f"{fam['title']}'s picture reader has no Metal kernels: it reads pictures on the CPU")
    if vision and not fam.get("vision"):
        S.fail(f"{fam['title']} has no image encoder")
    models_root = Path(a.models_dir) if a.models_dir else data / "models"
    folder = Path(a.gguf_dir) if a.gguf_dir else models_root / tag
    files = [folder / S.model_file(fam, model, i) for i in range(1, S.model_shards(fam, model)+1)]
    if a.gguf_dir and not a.update:
        for file in files:
            if not file.is_file():
                S.fail(f"missing local model: {file}")
    S.step(3, "Python packages")
    S.pip_install(S.requirement_lines() if S.REQUIREMENTS.exists() else S.PY_PACKAGES,
                  "numpy, jinja2, regex, cmake, ninja and server packages")
    S.step(4, "the strata-poly engine (Metal)")
    llama = S.llama_override()                         # None: no llama.cpp (STRATA_GGML_DIR overrides)
    if a.build:
        (BUILD_DIR / "BUILD-poly.json").unlink(missing_ok=True)
    exe = build_engine(llama, poly=True)
    if a.update:
        return 0

    def asset(path: Path, url: str) -> Path:
        if path.is_file() and (S.done(path) or S.whole_shard(path)):
            S.ok(f"using {path}")
        else:
            if a.no_download_models:
                S.fail(f"model downloads disabled; missing complete file: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            S.download(url, path)
        S.verify_sha256(path, *fam["sha256"][path.name])
        return path

    S.step(5, f"{fam['title']} {model}")
    base = d.get("hf", fam["hf"]).format(q=model)
    for file in files:
        asset(file, base + file.name)
    S.check_shards(files)
    draft = None
    if d.get("draft_model"):
        draft = folder / Path(d["draft_model"]).name
        draft = asset(draft, base + d["draft_model"])
    if d.get("draft") != "own" and fam.get("draft"):
        draft = S.find_in([data, *elsewhere], fam["draft"]) or data / fam["draft"]
        if not draft.is_file():
            if a.no_download_models:
                S.fail(f"model downloads disabled; missing draft: {draft}")
            S.run([sys.executable, str(ROOT.parent / "terminal" / "tools" / "glm_draft_fetch.py"), "fetch", "--out", str(draft)])
        S.run([sys.executable, str(ROOT.parent / "terminal" / "tools" / "glm_draft_fetch.py"), "verify", "--out", str(draft)])
    mmproj = vexe = None
    if vision:
        mmproj = asset(folder / fam["mmproj"], fam["mmproj_hf"] + fam["mmproj"])
        vexe = build_vision(llama)

    S.step(6, "preparing the tokenizer")
    pack = S.find_in([data, *elsewhere], f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    if not (pack / "tokenizer" / "vocab.json").exists():
        S.run([sys.executable, str(ROOT.parent / "terminal" / "tools" / "strata_tokenizer.py"), "--gguf", str(files[0]), "--out", str(pack)],
              env=dict(os.environ, **S.gguf_py_env()))
    profile = pack / "expert-profile.txt"
    shipped = d.get("profile", fam.get("profile"))
    if shipped and not profile.exists():
        pack.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT.parent / "terminal" / "data" / shipped, profile)
    args = ["--model", str(files[0]), "--max-context", str(ctx), "--expert-profile", str(profile)]
    if draft:
        args += ["--draft-model", str(draft)]
    if fam.get("tail_skip"):
        args += ["--tail-skip", str(fam["tail_skip"])]
    if info.get("perf_cores"):
        args += ["--threads", str(info["perf_cores"])]
    for flag, value in (("--conversation-cache-mib", a.conversation_cache_mib),
                        ("--vram-reserve-mib", a.vram_reserve_mib)):
        if value is not None:
            if value < 0:
                S.fail(f"{flag} cannot be negative")
            args += [flag, str(value)]
    cfg = mac_config(exe, args, pack, f"{fam['name']}-{model.lower()}", tag, a.port or 8080)
    cfg["display_name"] = f"{fam['title']} {model}"
    for key in ("host", "api_key"):
        if getattr(a, key):
            cfg[key] = getattr(a, key)
    if a.browser is not None:
        cfg["open_browser"] = a.browser
    if a.parallel is not None:
        cfg["parallel"] = max(1, a.parallel)
    if fam.get("one_request"):   # the engine keeps one sequence's state for it (setup.py writes the same)
        if (a.parallel or 1) >= 2:
            S.warn(f"--parallel {a.parallel}: {fam['title']} answers one request at a time on this engine")
        cfg["parallel"] = 1
    if vision:
        cfg["vision"] = {"exe": str(vexe), "mmproj": str(mmproj), "model": str(files[0]),
                         "gpu": not cpu_tower, "max_tokens": a.vision_tokens or fam.get("vision_most", 1024)}
        if cpu_tower:
            cfg["vision"]["threads"] = info.get("perf_cores") or max(1, (os.cpu_count() or 8) // 2)
    S.poly_kv_whole(cfg)
    cfg_path = config_path(tag)
    previous = P.read_config(cfg_path)
    P.apply_options(cfg["args"], a, previous)
    headers, why = P.read_headers([*files, *([draft] if draft else [])])
    if getattr(a, "poly_kv", None) == "q8_0" and (
            not headers or headers[0].metadata.get("general.architecture") != "qwen35moe"):
        S.fail("--poly-kv q8_0 requires a verified Qwen35MoE GGUF header")
    estimate = P.estimate(headers, ctx, kv=P.flag_value(cfg["args"], "--kv", "auto"),
                          memory=P.flag_value(cfg["args"], "--metal-memory", "balanced"),
                          parallel=cfg.get("parallel", previous.get("parallel", 1)))
    if why:
        estimate["reason"] = why
    P.report(estimate, S.say)
    if P.flag_value(cfg["args"], "--metal-memory") == "balanced":
        S.say("  balanced KV grows in 4096-token steps with 1024 tokens ahead, up to the chosen context")
    S.write_setup_config(cfg_path, cfg)
    script = S.write_run_script(tag, cfg_path, cfg["port"], cfg.get("open_browser") is True)
    S.ok(f"start script: {script.name}")
    return 0 if a.no_start else start(cfg_path, a.port, a.browser is not False)


def setup_q36(a, info: dict, data: Path, elsewhere: list) -> int:
    """The separate Qwen driver, using the same pinned catalog and server schema."""
    fam = S.OWN_PATH_FAMILIES[a.family]
    model = a.model or S.own_path_size(fam, info["ram_gb"])
    if model not in fam["models"]:
        S.fail(f"{fam['title']} has no {model} file", "choose one of: " + ", ".join(fam["models"]))
    ctx = a.context or 8192
    if ctx <= 0:
        S.fail("--context must be positive")
    S.warn("strata-q36 on Metal is experimental; for Qwen3.6 on a Mac the qwen36 family (strata-poly) is the one to "
           "install (docs/MAC.md)")
    tag = fam["tag"] + model
    root = Path(a.models_dir) if a.models_dir else data / "models"
    folder = Path(a.gguf_dir) if a.gguf_dir else root / tag
    gguf = folder / fam["file"].format(q=model)
    vision = a.vision == "yes"
    assets = [(gguf, fam["hf"])]
    if vision:
        assets.append((folder / fam["mmproj"], fam["mmproj_hf"]))
    # Fail before building or fetching dependencies for a missing offline asset.
    if not a.update and (a.gguf_dir or a.no_download_models):
        for path, _ in assets:
            if not path.is_file():
                S.fail(f"missing local model asset: {path}")
    S.pip_install(S.requirement_lines() if S.REQUIREMENTS.exists() else S.PY_PACKAGES,
                  "numpy, jinja2, regex, cmake, ninja and server packages")
    llama = S.llama_override()                         # None: no llama.cpp (STRATA_GGML_DIR overrides)
    if a.build:
        (BUILD_DIR / "BUILD-q36.json").unlink(missing_ok=True)
    exe = build_engine(llama, engine="strata-q36")
    if a.update:
        return 0
    for path, base in assets:
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            S.download(base + path.name, path)
        S.verify_sha256(path, *fam["sha256"][path.name])
    pack = S.find_in([data, *elsewhere], f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    if not (pack / "tokenizer" / "vocab.json").is_file():
        S.run([sys.executable, str(ROOT.parent / "terminal" / "tools" / "strata_tokenizer.py"), "--gguf", str(gguf), "--out", str(pack)],
              env=dict(os.environ, **S.gguf_py_env()))
    args = ["--native", str(gguf), "--max-context", str(ctx), "--prefill", "auto", "--mtp", str(fam["mtp"])]
    if info.get("perf_cores"):
        args += ["--pool-workers", str(info["perf_cores"])]
    if a.vram_reserve_mib is not None:
        if a.vram_reserve_mib < 0:
            S.fail("--vram-reserve-mib cannot be negative")
        args += ["--expert-vram", str(max(0, info["ram_gb"] - 3 - a.vram_reserve_mib / 1024))]
    cfg = mac_config(exe, args, pack, f"{fam['name']}-{model.lower()}", tag, a.port or 8080)
    cfg["arch"] = fam["arch"]
    cfg["display_name"] = f"{fam['title']} {model}"
    for key in ("host", "api_key"):
        if getattr(a, key):
            cfg[key] = getattr(a, key)
    if a.browser is not None:
        cfg["open_browser"] = a.browser
    if a.parallel is not None:
        cfg["parallel"] = max(1, a.parallel)
    if vision:
        cfg["args"].append("--vision")
        cfg["vision"] = {"exe": str(build_vision(llama)), "mmproj": str(folder / fam["mmproj"]),
                         "model": str(gguf), "gpu": True, "max_tokens": a.vision_tokens or 1024}
    cfg_path = config_path(tag)
    S.write_setup_config(cfg_path, cfg)
    script = S.write_run_script(tag, cfg_path, cfg["port"], cfg.get("open_browser") is True)
    S.ok(f"start script: {script.name}")
    return 0 if a.no_start else start(cfg_path, a.port, a.browser is not False)


# ------------------------------------------------------------------------------------------------ the DFlash2 drafter
def fetch_drafter(model_name: str, data: Path) -> Path | None:
    """The DFlash2 drafter of a set whose Mac default is DFlash2 (strata_control/drafting.py SET_DEFAULTS: UD-Q4_K_XL and UD-IQ4_XS-v31): the tuned one (0.34 GB), else the stock one
    (1.2 GB), through the normal download path (resumable, the pinned size and SHA-256). Offline, or an address that does not answer: nothing is said and the model keeps the MTP
    (owner, strata-offline-users). A fault that is the person's to know (a file that failed its safety check) is one plain line. Never stops the setup. None when it is not here."""
    from strata_control import acquisition, drafting, drafting_fetch
    if drafting.choice_of("auto", model_name=model_name, backend="metal") != "dflash2":
        return None
    said = []

    def show(step: str, done: int, total: int) -> None:
        if not said:                                       # the line appears when the download starts: a computer with no internet sees nothing
            said.append(True)
            gb = drafting.TUNED_GB if "tuned" in step else drafting.DOWNLOAD_GB
            S.say(f"  Downloading the DFlash2 drafter ({gb:g} GB, once) ...")
    try:
        got = drafting_fetch.ensure_for(data, model_name=model_name, backend="metal", progress=show)
    except (drafting_fetch.DraftingError, acquisition.AcquisitionRefused, OSError) as exc:
        S.warn(f"{exc} This model keeps using the MTP.")
        return None
    if got is not None:
        S.ok(f"DFlash2 drafter: {got}")
    return got


# ------------------------------------------------------------------------------------------------ main
def _feed_models(argv, interactive: bool) -> None:
    """The signed feed's models (setup.py feed_models_setup: the feed, else the last verified list) join MAC_MODELS / FAMILIES / MODELS. Read when a name the built-in
    tables do not know is given, or when the size menu is going to be shown; nothing happens when the feed has no models."""
    args = list(argv or sys.argv[1:])
    peek = lambda flag: next((v.split("=", 1)[1] if v.startswith(flag + "=") else args[i + 1] if i + 1 < len(args) else None   # noqa: E731
                              for i, v in enumerate(args) if v == flag or v.startswith(flag + "=")), None)
    family, model = peek("--family"), peek("--model")
    if (family and family not in ("qwen", "unsloth", "gyro")) or (model and model not in MAC_MODELS) or interactive:
        S.feed_models_setup(peek("--feed"))


def main(argv=None) -> int:
    given = argv if argv is not None else sys.argv[1:]
    _feed_models(argv, not any(x in given for x in ("--yes", "--model", "--check", "--family")) and ("--setup" in given or not mac_configs()))   # (a plain start of an installed model reads nothing)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list-models", action="store_true",
                    help="list all model families and quantizations from the shared installer")
    ap.add_argument("--family", choices=list(model_catalog()), default=None,
                    help="model family (quantization names are scoped to this family)")
    ap.add_argument("--model", help="quantization from the chosen family (see --list-models)")
    ap.add_argument("--no-download-models", action="store_true", help="use existing GGUF, draft and projector files only")
    ap.add_argument("--parallel", type=int)
    ap.add_argument("--vision-tokens", type=int)
    ap.add_argument("--conversation-cache-mib", type=int, default=None)
    ap.add_argument("--vram-reserve-mib", type=int, default=None)
    ap.add_argument("--context", type=int)
    ap.add_argument("--port", type=int)
    ap.add_argument("--host")
    ap.add_argument("--api-key")
    ap.add_argument("--draft-vocab", choices=list(S.DRAFT_VOCABS))
    ap.add_argument("--no-browser", dest="browser", action="store_false", default=None)
    ap.add_argument("--data-dir")
    ap.add_argument("--move-data", action="store_true", help="with --data-dir: move the model files already here into the new folder")
    ap.add_argument("--set-default-data-dir", action="store_true", help="with --data-dir: remember the new folder without moving anything")
    ap.add_argument("--models-dir")
    ap.add_argument("--gguf-dir")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--source", choices=S.SOURCES, default=None,
                    help="where model files come from: auto (huggingface), huggingface, modelscope; "
                         "same as STRATA_SOURCE")
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--no-start", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--build", action="store_true", help="build the engine again even when it is current")
    ap.add_argument("--vision", choices=["yes", "no", "gpu", "cpu"],
                    help="let the model read images (the image encoder, built here; yes = gpu: it reads on the Mac's GPU, cpu is ~100x "
                         "slower; default: asked, yes)")
    P.add_arguments(ap)
    a, rest = ap.parse_known_args(argv)
    if a.list_models:
        for family, entry in model_catalog().items():
            S.say(f"{family}: {entry['title']} — {entry['metal_engine']} Metal")
            for model, spec in entry["models"].items():
                size = f" ({spec['download_gb']:.1f} GB download)" if "download_gb" in spec else ""
                S.say(f"  {model}{size}")
        return 0
    try:
        P.validate(a, MAC_POLY_FAMILIES)
    except ValueError as e:
        ap.error(str(e))
    if a.host and a.host not in ("127.0.0.1", "localhost", "::1") and not a.api_key:
        S.fail("a non-loopback --host requires --api-key")
    if rest:
        S.warn("not used on a Mac: " + " ".join(rest))
    # setup.py hands a Mac over before its own parser runs, so --source has to be honoured here too:
    # source_hint() recommends it on every platform, and a Mac that dropped it failed the download
    # with the same unreachable host (#2363).
    if a.source:
        os.environ["STRATA_SOURCE"] = a.source
    S.say("PolyStrata on a Mac (Apple Silicon, Metal)")
    data, elsewhere = S.data_folder_for(a)
    roots = [data, *elsewhere]
    models_root = Path(a.models_dir) if a.models_dir else data / "models"

    have = mac_configs()
    if have and not (a.setup or a.family or a.model or a.check or a.no_start or a.update or a.build or P.requested(a)):
        return start(have[0], a.port, a.browser is not False)

    # ---- 1. the Mac
    S.step(1, "checking your Mac")
    info = mac_info()
    S.ok(f"{info['chip']}, {info['ram_gb']:.0f} GB unified memory, macOS {info['macos'] or '?'}")
    problems = mac_problems(info)
    for p in problems:
        S.warn(p)
    free = S.free_gb(data if data.exists() else ROOT)
    S.ok(f"free disk: {free:.0f} GB at {data}")
    if a.check:
        return 1 if problems else 0
    if problems:
        S.fail(problems[0])

    if a.family in S.OWN_PATH_FAMILIES:
        return setup_q36(a, info, data, elsewhere)
    if a.family in MAC_POLY_FAMILIES:
        return setup_poly(a, info, data, elsewhere)
    if a.model and a.model not in MAC_MODELS:
        S.fail(f"{a.model} needs its smaller-model --family; see --list-models")

    # ---- 2. the choices
    S.step(2, "your choices")
    rec, notes = recommend_model(info["ram_gb"])
    for n in notes:
        S.warn(n)
    sizes = S.family_models(a.family) if a.family else MAC_MODELS
    if a.model and a.model not in sizes:
        S.fail(f"{a.family} has no {a.model} file", "choose one of: " + ", ".join(sizes))
    if rec not in sizes:
        rec = min(sizes, key=lambda m: S.MODELS[m]["download_gb"])
    if a.model:
        model = a.model
    else:
        names = list(sizes)   # --model offers the others
        for i, m in enumerate(names, 1):
            S.say(f"  {i}) {m:8} {S.MODELS[m]['about']}  (maps {model_metadata(m, a.family)['shard1_gb']:.0f} GB"
                  + ("" if model_metadata(m, a.family)["tested"] else ", not measured on a Mac") + ")"
                  + ("  (recommended)" if m in RECOMMENDED else "") + ("   <- default" if m == rec else ""))
        model = names[int(S.ask("Which size?", [str(i) for i in range(1, len(names) + 1)],
                                str(names.index(rec) + 1), a.yes)) - 1]
    for n in model_note(model, info["ram_gb"], a.family):
        S.warn(n)
    vision = a.vision or ("yes" if S.ask("Should the model also read images? (the image encoder, ~1 GB more)",
                                          ["y", "n"], "y", a.yes) == "y" else "no")   # on by default (strata_control/vision_reader.py)
    if vision == "cpu":
        from strata_control import vision_reader as VR
        S.warn(VR.MAC_CPU_NOTE)
    vision_gpu = vision != "cpu"
    vision = "no" if vision == "no" else "yes"
    ctx = a.context or recommend_context(info["ram_gb"])
    if ctx > TESTED_CONTEXT:
        S.say(f"  context {ctx}: measured on the Mac up to {TESTED_CONTEXT} (needles found at 8K-32K); longer is kept")
    S.ok(f"{model}, context {ctx}")
    family = a.family or MAC_MODELS[model].get("family", "qwen")
    fam = S.FAMILIES[family]
    tag = fam["tag"] + model
    need = (0 if a.gguf_dir else S.MODELS[model]["download_gb"]) + 6   # --gguf-dir: the files are there already
    if free < need:
        S.warn(f"about {need:.0f} GB are needed on the disk ({free:.0f} GB free): kept as you chose; it may run out")

    # ---- 3. Python packages (cmake and ninja come with the build in step 4, from requirements-build.txt)
    S.step(3, "Python packages")
    S.pip_install(S.requirement_lines() if S.REQUIREMENTS.exists() else S.PY_PACKAGES, "numpy, jinja2, ...")

    # ---- 4. the engine
    S.step(4, "the PolyStrata engine (Metal)")
    llama = S.llama_override()                         # None: ggml is the vendored third_party/strata-ggml (STRATA_GGML_DIR overrides)
    S.ok(f"llama.cpp {S.LLAMA_CPP_COMMIT[:7]} (ggml, gguf-py)")
    if a.build:
        (BUILD_DIR / "BUILD.json").unlink(missing_ok=True)
    exe = build_engine(llama)
    if a.update:
        # update.sh's own text: "The model files are not touched".  The questions are asked anyway, so when the
        # answers name a model (or an image encoder) that is not here, say it: a silent exit reads like a broken
        # download (an M1 Max chose IQ3_S in update.sh and got no model and no message).
        files = [Path(a.gguf_dir) / n if a.gguf_dir else models_root / tag / n for n in model_shard_names(model)]
        pack = S.find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
        pack_done = (pack / "native_experts.txt").exists() and (pack / "tokenizer" / "vocab.json").exists()
        if not (all(f.exists() and S.done(f) for f in files) and pack_done):
            S.warn(f"{model} is not installed here, and an update does not download model files: nothing was installed")
            S.say(f"  to download and prepare {model}: ./setup.sh --setup")
        elif vision == "yes" and not (models_root / fam["mmproj"]).exists():
            S.warn("the image encoder is not downloaded here, and an update does not download model files")
            S.say("  to add image support: ./setup.sh --setup")
        return 0

    # ---- 5. the model
    S.step(5, f"downloading {fam['title']} {model}")
    models_dir = Path(a.gguf_dir) if a.gguf_dir else models_root / tag
    models_dir.mkdir(parents=True, exist_ok=True)
    ready = fam.get("ready", False)
    if ready:
        for asset in S.ready_files(model):
            path = S.ready_where(models_dir, asset)
            if path.is_file():
                S.verify_sha256(path, asset[4], asset[5])
            elif a.no_download_models:
                S.fail(f"model downloads disabled; missing ready-pack file: {path}")
        if not a.no_download_models:
            S.ready_fetch(model, models_dir)
        rp = S.ready_paths(model, models_dir)
        files = [rp["dense"], rp["ple"]]
    else:
        files = [models_dir / n for n in model_shard_names(model, family)]
    for s in ([] if ready else files):
        if s.exists() and (S.done(s) or a.no_download_models and S.whole_shard(s)):
            S.ok(f"{s.name} already downloaded")
            continue
        if a.no_download_models:
            S.fail(f"model downloads disabled; missing complete file: {s}")
        S.download(S.model_hf(fam, model) + s.name, s)
    S.check_shards(files)
    for s in files:                                    # the Unsloth files: pinned sizes and SHA-256
        if (family == "unsloth" or fam.get("feed")) and s.name in fam.get("sha256", {}):       # the family's pins, and UD-Q4_K_XL's second layout by the size on the disk
            S.verify_sha256(s, *S.fam_pin(fam, s), own=bool(a.gguf_dir))
        elif shard_sha(s.name):
            S.verify_sha256(s, *shard_sha(s.name))
    S.ok("model files present")

    mmproj = vexe = None
    if vision == "yes":
        mmproj = models_root / fam["mmproj"]
        if not (mmproj.exists() and (S.done(mmproj) or a.no_download_models and S.whole_shard(mmproj))):
            if a.no_download_models:
                S.fail(f"model downloads disabled; missing projector: {mmproj}")
            S.download(fam["mmproj_hf"] + fam["mmproj"], mmproj, "vision encoder")
        sha = VISION_SHA256.get(fam["mmproj"])
        if sha:
            S.verify_sha256(mmproj, *sha)
        vexe = build_vision(llama)

    # ---- 6. the pack and the MTP draft layer
    S.step(6, "preparing the model for PolyStrata")
    env = dict(os.environ, **S.gguf_py_env())
    if ready:
        pack = rp["pack"]
        shards = files
    else:
        pack = S.find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
        # The repack and the pack move 100+ GB through the disk and the pack's python holds up to 20 GB: no engine may run beside
        # them (an engine of 70-90 GB with a repack froze the 128 GB M5 Max twice).  A lock beside the GPU's file tells the engine
        # and start() to wait; a running engine stops setup here.
        repack = family == "swift" or len(files) > 2
        lock_dir = data / "metal" if repack else files[0].parent
        native0 = data / "metal" / f"{tag.lower()}-metal.gguf"
        work = (repack and not (native0.exists() and S.done(native0))) or \
            not ((pack / "native_experts.txt").exists() and (pack / "tokenizer" / "vocab.json").exists())
        prep_lock = lock_dir / PREPARE_LOCK
        if work:
            running = engine_pids()
            if running:
                S.fail(f"a PolyStrata engine is running (pid {running[0]}): preparing a model beside it can freeze this Mac",
                       "stop the server (close its window), then run ./setup.sh --setup again")
            lock_dir.mkdir(parents=True, exist_ok=True)
            prep_lock.write_text(str(os.getpid()))
            atexit.register(lambda: prep_lock.unlink() if prep_lock.exists() else None)
        if repack:
            # many shards: the experts and dense tensors into one GGUF the GPU maps (or streams), the n-gram table read
            # from its own shard (every tensor checked byte for byte: --verify)
            native = data / "metal" / f"{tag.lower()}-metal.gguf"
            native.parent.mkdir(parents=True, exist_ok=True)
            if not (native.exists() and S.done(native)):
                S.say(f"  Writing the model's experts and weights into one file for the GPU (one time, "
                      f"{model_metadata(model, family)['shard1_gb']:.0f} GB) ...")
                S.run([sys.executable, str(ROOT / "metal" / "tools" / "metal_repack.py"), "--shard1", str(files[0]),
                       "--out", str(native), "--verify"])
                S.mark(native)
            ple = ple_shard(files)
            shards = [native, ple]
        elif family == "gyro":
            shards = [files[0], files[0]]                  # one file: the model and its n-gram table (--native, --ple-gguf)
        else:
            shards = files
        if not (pack / "native_experts.txt").exists() or not (pack / "tokenizer" / "vocab.json").exists():
            S.run([sys.executable, str(ROOT.parent / "terminal" / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
                   *fam.get("pack_args", [])], env=env)   # Unsloth: --compat-bf16 (the host's copies; the GPU reads the file)
        S.ok(f"model prepared: {pack}")
        prep_lock.unlink(missing_ok=True)
    mtp = (S.find_in(roots, "mtp/mtp-q2_0.gguf") or data / "mtp" / "mtp-q2_0.gguf").parent
    rt = mtp / "rt"
    stream = ready or streams(model, info["ram_gb"], family)
    use_mtp = stream_mtp(stream, info["ram_gb"])
    if stream and not use_mtp:
        S.ok("the experts will stream from the SSD: no MTP draft layer (one token per step is faster there); "
             "./setup.sh --setup on a Mac with more memory adds it")
    elif stream:
        S.ok(f"the experts will stream from the SSD; the MTP draft layer stays on, with --spec {STREAM_SPEC} "
             "(a longer window pays when every window has a fixed cost per layer)")
    if stream and not ready and info["ram_gb"] < STREAM_MTP_FROM_GB:
        experts_file(shards[0])           # a small Mac: its misses are many; a big Mac's arena holds most experts
    if use_mtp and not (mtp / "mtp-q2_0.gguf").exists():
        if a.no_download_models:
            S.fail(f"model downloads disabled; missing draft: {mtp / 'mtp-q2_0.gguf'}")
        S.say("  The MTP draft layer (speculative decoding) comes from the original Qwen checkpoint: only its ~5 GB of")
        S.say("  MTP tensors are downloaded.")
        S.run([sys.executable, str(ROOT.parent / "terminal" / "tools" / "mtp_fetch.py"), "fetch", "--out", str(mtp)],
              env={**env, "STRATA_SOURCE": S.model_source()})
        S.run([sys.executable, str(ROOT.parent / "terminal" / "tools" / "mtp_pack.py"), "--src", str(mtp), "--experts", "q2_0",
               "--out", str(mtp / "mtp-q2_0.gguf")], env=env)
    if use_mtp:
        rt.mkdir(parents=True, exist_ok=True)          # the Metal engine reads mtp-q2_0.gguf beside it (no rt pack)
        S.refresh_draft_vocab(rt, a.draft_vocab or "cjk")
        S.ok(f"MTP draft layer: {mtp / 'mtp-q2_0.gguf'}")

    fetch_drafter(f"{fam['name']}-{model.lower()}", pack.parent.parent)       # DFlash2 is this set's default drafter (the MTP above stays as the fallback); the folder a start looks in

    # ---- 7. the config, the start script
    S.step(7, "writing the start script")
    port = a.port or 8080
    cfg = mac_config(exe, engine_args(pack, shards, rt, ctx, stream, info["ram_gb"], model), pack, f"{fam['name']}-{model.lower()}", tag, port)
    if ready:
        cfg["args"] += ["--native-dense-gguf", str(rp["dense"])]
        if "embd" in rp:                               # a prepared pack has the token embedding in its dense GGUF
            cfg["args"] += ["--embd-gguf", str(rp["embd"])]
    if fam.get("profile"):
        cfg["args"][cfg["args"].index("--expert-profile") + 1] = str(ROOT.parent / "terminal" / "data" / fam["profile"])
    if a.parallel is not None:
        cfg["parallel"] = max(1, a.parallel)
    cfg["env"] = {"STRATA_METAL_STREAM": "1" if stream else "0"}   # the engine follows setup's measured choice (STRATA_METAL_F16 is the engine's own default)
    if vision == "yes":
        cfg["args"] += ["--vision"]
        cfg["vision"] = {"exe": str(vexe), "mmproj": str(mmproj), "model": str(files[0]), "gpu": vision_gpu,
                         "max_tokens": a.vision_tokens or 1024}
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.draft_vocab:
        cfg["draft_vocab"] = a.draft_vocab
    if a.browser is not None:
        cfg["open_browser"] = a.browser
    cfg_path = config_path(tag)
    S.write_config(cfg_path, cfg)
    script = S.write_run_script(tag, cfg_path, port, cfg.get("open_browser") is True)
    S.ok(f"start script: {script.name}")
    S.say()
    S.say("All set.")
    S.say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1")
    S.say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    S.say(f"  Next time:        just run ./setup.sh (or {script.name}) - it starts right away")
    if a.no_start:
        return 0
    return start(cfg_path, port, a.browser is not False)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        S.say("\nstopped.")
        sys.exit(1)
