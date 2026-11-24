# Is PolyStrata safe and private?

This page says what PolyStrata downloads, where from, when it uses the internet, what stays on your PC and how the downloads
are checked. Everything here comes from reading the code of this version. Each section names the files, so you or anyone
you trust can check it. Where PolyStrata does less than you might hope, the page says so.

## At a glance

- **Your prompts, files, pictures and models stay on your PC.** They leave only when you choose it: you open the server to
  other devices or to the internet, or you add a remote tool server.
- **There is no telemetry.** PolyStrata sends no usage data, no crash reports and no analytics, and has no account to sign in to.
- **Starting and running a model never downloads anything.** Downloads and updates are steps that you start.
- **The server listens on your PC only.** `strata launch` makes an API key for you when you share it with other devices.
- **Once a model is downloaded, PolyStrata runs without internet.** Starting, loading and serving need no connection. Only
  things you ask for use it: the tunnel, remote tool servers you add, and installs and updates.
- **Downloads are checked.** Updates come from a signed feed. Every model file has a fixed SHA-256 and size, and an install
  stops if the pinned version of a file is gone (the details are [below](#how-downloads-are-checked)).
- **The code is open.** Anyone can read it, and security problems go to [SECURITY.md](../.github/SECURITY.md).

## Does PolyStrata phone home?

No. A search of the Terminal, the engine server and the installers found no analytics library, no crash reporter and no
code that reports usage anywhere.

One thing sounds similar and stays on your PC: the Monitor, `/metrics` and `/v1/telemetry` show hardware readings
(temperature, speed, memory) to you. They are served from your own PC, and they need your API key when you set one.

Like any download, each connection below shows the other side your IP address and the file you asked for.

## When PolyStrata uses the internet

| What | Where it goes | When |
| --- | --- | --- |
| Model files, the picture reader, the speed-up layer (MTP) | huggingface.co, or the mirror in `HF_ENDPOINT` if you set one | When you install a model |
| The same files from ModelScope, for PCs that cannot reach Hugging Face | modelscope.cn | Only if you ask for it (`--source modelscope`). PolyStrata never switches to it by itself |
| Engine, Terminal and Control updates | The signed feed on GitHub Releases of Niko1221/Strata | When you install an engine or run `strata update` |
| llama.cpp source code (a fixed commit) | github.com/ggml-org/llama.cpp | During a model install, if it is not on your PC yet. PolyStrata uses only its model-preparation scripts |
| Cloudflare tunnel | Cloudflare, through the `cloudflared` program | Only while you have the tunnel on, and `strata launch` makes an API key for it unless you pass `--no-api-key` (then it warns that anyone who finds the address can use your model) |
| Tailscale, local network | Not PolyStrata's traffic. PolyStrata only reads Tailscale's status and shows the address | Only if you choose that access in `strata launch` |
| An image link in an API request | The address in the link | Only when a program sends your server a request with an `http(s)` image link. `--offline` refuses it |
| Remote tool servers (MCP) | The address you put in the run config | Only when you add one |

That is the full list found in the code. Some details:

- **An engine is never updated by itself.** The checks in the table run when you run a command. With no connection, the check
  says so and nothing else happens. The Terminal fetches an engine by itself only when none is installed (and `strata launch`
  asks first when the one installed is older than the model needs). `strata launch`, setup and a model download never fetch,
  replace or remove an engine that is installed. An update is `strata update`: with a model loaded it names the model, asks
  before stopping it, and downloads while the model keeps serving; a script is refused (exit 7) unless it passes
  `--stop-model`. One quiet exception, the launch monitor only: once a day it reads the signed update feed (a few KB, no
  engine, a 5 second limit, nothing shown if it fails or there is no connection) so it can say "a new engine is available".
  `STRATA_UPDATE_CHECK=0` turns it off (`terminal/strata_control/engine_update.py`, `refresh_cache`).
- **Starting and serving never download.** `strata serve` fetches nothing. Setup starts the server with
  `STRATA_NO_ACQUIRE=1`, and the downloaders refuse to run under it (`terminal/strata_control/acquisition.py`).
- **The engine itself has no network code.** The part of PolyStrata that runs the model (`engine/src/`, `engine/include/`,
  `engine/sycl/`, `engine/metal/`) contains no networking. The server's web page loads no fonts, scripts or trackers from the
  internet.
- **Building from source** reaches more places. Python from python.org or winget (`START-HERE.bat`), a standalone Python
  from GitHub on a Mac (`setup.sh`), Python packages from PyPI (`terminal/requirements.txt`, pinned versions), NVIDIA's CUDA libraries
  from PyPI and, on Ubuntu and Debian, NVIDIA's apt key and packages, AMD's ROCm wheels, the Vulkan SDK from LunarG, and
  llama.cpp and stable-diffusion.cpp source from GitHub. These run only while you run the setup.

## What stays on your PC

- **Prompts, attached files and pictures** are handled by the engine on your PC. Models are in the folder you chose.
- **The server listens on `127.0.0.1`**, this PC only. That is the engine server's default (`terminal/serve/server.py`, the
  `--host` option).
- **Sharing needs a key.** `strata launch` makes a key for you when you pick anything wider than this computer, and
  `--no-api-key` shares without one only with a warning (`terminal/strata_cli/launch/access.py`). The bare engine server
  (`terminal/serve/server.py`) only prints a warning if you open it to your network without a key, so set one yourself in
  that case.
- **Programs you connect** (Linger, Claude Code, Codex, and so on) are separate programs with their own network behaviour.
  PolyStrata does not control what they send.

The server also has browser protections (a host check, an origin check, no CORS by default). They are in
[SECURITY.md](../.github/SECURITY.md#what-the-server-exposes).

## How downloads are checked

**Updates for the engine, the Terminal and Control.**

1. The release publishes `strata-feed.json` and `strata-feed.json.sig`, signed with an Ed25519 key. The matching public key is
   inside every installed PolyStrata (`strata_control/update_keys.json`). The private key is kept offline by the owner.
2. Before it uses the feed, your PC checks the signature, that the feed has not expired and that it is not older than the
   last feed it accepted (so an old feed cannot be replayed).
3. Each engine package is checked against its SHA-256 and its signed manifest, then every file in it against its own hash,
   then it is checked to fit your PC.
4. The new engine is test-started before PolyStrata switches to it. If anything fails, you keep the engine you had.

Updates never replace your models or saved tuning. The full description is in [packaging/UPDATES.md](../tools/packaging/UPDATES.md).

**Model files.** Every model file that `setup.py` downloads has a fixed SHA-256 and size, written in the code, so the check
needs no network: the main Qwen3.8-Flash-Next files (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S), Swift 1.5, the Coder, the Unsloth sizes,
the Gyro files, Qwen3.6, the NVFP4 packs, the picture reader, the image model files and the speed-up layer (MTP, tensor by
tensor). A file that does not match is deleted and the install stops. A file whose size on the server is not the pinned
size stops it before the download starts.

- If the pinned version of a file is gone from Hugging Face, the install stops and says to update PolyStrata. It does not take
  the repository's current file instead. A tester can set `STRATA_ALLOW_UNPINNED=1` to take the current file anyway, without
  the check. It is off by default, and setup says so when it is set.
- Starting a model does not check its files, and running setup again for a model you have does not download or hash them
  again. A file installed before these hashes existed is kept as it is.
- Files from ModelScope are checked against the same fixed SHA-256. ModelScope serves a repository's current files, so a file
  it has changed fails the check. A file without a pin is checked against the hash ModelScope publishes, and setup warns
  when it publishes none.
  (`setup.py`: `REPO_PINS`, `download`, `verify_sha256`; `terminal/tools/mtp_fetch.py`.)

**Engine in the source route.** `setup.py` checks the engine zip against the SHA-256 that GitHub's API publishes for that
release file. If GitHub cannot say (offline, rate limit), setup warns and installs anyway. `STRATA_SKIP_SHA256=1` switches the
check off, and it is never on by default.

**Getting PolyStrata.** It comes from a git clone or the source zip of `simplexai-labs/PolyStrata`, so the first install
trusts HTTPS and that repository ([SECURITY.md](../.github/SECURITY.md#installing-from-this-repository)).
The Strata project's one-line Terminal install is not part of PolyStrata.

**What the checks do not cover.** They catch a damaged or swapped download and a feed nobody signed with the release key. They
cannot protect you if someone who can publish a release (or who holds the signing key) publishes something harmful.

## Open source and reporting a problem

PolyStrata is MIT-licensed and the whole code is at
[github.com/simplexai-labs/PolyStrata](https://github.com/simplexai-labs/PolyStrata). Read it, build it and compare.
PolyStrata has not had an outside security audit yet.

To report a security problem, use GitHub's private reporting (the repository's **Security** tab), as described in
[SECURITY.md](../.github/SECURITY.md). Ordinary questions and ideas can go to an
[issue](https://github.com/simplexai-labs/PolyStrata/issues).

## Where to check this in the code

| Topic | Files |
| --- | --- |
| Model downloads and pins | `terminal/setup.py`, `terminal/tools/mtp_fetch.py`, `terminal/tools/image/qimage.py` |
| Update feed, signatures | `terminal/strata_control/feed.py`, `signing.py`, `transfer.py`, `update_keys.json`, `terminal/strata_cli/update_cmd.py` |
| Downloads refused while running | `terminal/strata_control/acquisition.py` |
| One-line installers | `tools/packaging/install.sh`, `tools/packaging/install.ps1` |
| Sharing, tunnel | `terminal/strata_cli/launch/access.py` |
