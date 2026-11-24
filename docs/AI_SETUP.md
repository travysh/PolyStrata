# Installing PolyStrata for a user (instructions for AI agents)

This page is written for an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...) that a user asked to
set up PolyStrata on their PC. PolyStrata runs large AI models locally (Qwen3.8-Flash-Next by default, and eleven
others) on one NVIDIA or AMD graphics card plus system RAM, and serves an OpenAI- and Anthropic-compatible API on
`http://127.0.0.1:8080`. The human-oriented version of everything below is [INSTALL.md](INSTALL.md).

Work through the steps in order. Tell the user what you are doing in plain words; they may not be technical.

Setup runs from the PolyStrata folder: `START-HERE.bat` on Windows, `./setup.sh` on Linux and on a Mac. For Apple
Silicon, `./setup.sh --list-models` lists the complete catalog; read [Mac model coverage](MAC.md#model-coverage) before
you pick a model there, since several have not run on a Mac with their real weights.

## 0. Ground rules

- **Never expose the server without an API key.** Keep the default `--host 127.0.0.1`. Only if the user asks for access
  from other devices, use `--host 0.0.0.0` **together with** `--api-key <a long random secret>`, and give the user the
  key. Never put a tunnel or port forward in front of a server without a key.
- Do not change the user's system beyond what setup does. Setup installs Python for the user account if needed, and
  everything else inside the PolyStrata folder and its `PolyStrata-data` folder. Ask before installing drivers.
- The model download is 70 to 120 GB, depending on the model. Confirm the user is fine with that before you start,
  especially on a metered connection.
- Setup is long-running (an hour or more on a slow connection). Run it in the background or with a long timeout and
  poll its output; do not kill it because it is quiet for a while. It is resumable: running the same command again
  continues where it stopped.

## 1. Check the PC

The quickest way is setup's own check, which prints the GPU or GPUs, driver, RAM, CPU, free disk and whether this PC can
run PolyStrata. It installs nothing except Python and the `terminal/.venv` it needs to run:

```
Windows:  START-HERE.bat --check
Linux:    ./setup.sh --check
```

To check by hand:

| What | Windows (PowerShell) | Linux |
| --- | --- | --- |
| OS | `[Environment]::OSVersion` | `cat /etc/os-release` |
| NVIDIA GPU, VRAM, driver | `nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv` | the same |
| AMD GPU | `Get-CimInstance Win32_VideoController \| Select-Object Name, AdapterRAM` (AdapterRAM is capped at 4 GB; trust the model name) | `lspci \| grep -i -E 'vga\|display'`; VRAM: `cat /sys/class/drm/card*/device/mem_info_vram_total` (bytes) |
| RAM | `(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB` | `free -g` |
| Free disk | `Get-PSDrive -PSProvider FileSystem` | `df -h .` |

Requirements (details: [INSTALL.md](INSTALL.md#what-you-need)):

- **GPU:** NVIDIA RTX 20, 30, 40 or 50 series, or an AMD Radeon card listed in [AMD_HIP.md](AMD_HIP.md), with 12 GB of
  VRAM or more (an NVIDIA card with 8 GB runs, slowly). GTX 10 series and older, and integrated GPUs, are not supported.
  Older cards have experimental paths the user opts into ([OLDER_GPUS.md](OLDER_GPUS.md)).
- **Driver:** NVIDIA 580 or newer for the default engine. AMD on Linux: the kernel's amdgpu driver; on Windows: a current
  AMD Software driver. If the driver is missing or too old, tell the user to update it and restart; do not install
  drivers yourself unless they ask.
- **RAM:** 32 GB or more (see step 3). **Disk:** 70 to 120 GB free, ideally on an NVMe SSD. **CPU:** x86-64 with AVX2.
  Without AVX2, setup still installs, as an experimental and slow build it compiles on the PC (10 to 20 minutes). Tell the
  user that before starting ([INSTALL.md](INSTALL.md#older-cpus-experimental)).
- **OS:** Windows 10/11 or Linux. Ubuntu 22.04/24.04 and Debian 12/13 are fully automatic.

Model support is separate from hardware support: each model's own needs are in step 3. If the PC does not meet them, say which part is missing and stop.

## 2. Get PolyStrata

```
git clone https://github.com/simplexai-labs/PolyStrata.git
```

Without git: download https://github.com/simplexai-labs/PolyStrata/archive/refs/heads/main.zip and unzip it. Pick a drive with
enough free space: the model files go to a `PolyStrata-data` folder **next to** the PolyStrata folder (or `--data-dir <path>`
on another drive).

## 3. Pick the model

By the PC's RAM. Ask the user whether they mainly want it for code, because the Coder is a good pick at 32 GB:

| RAM | `--family` / `--model` | Notes |
| --- | --- | --- |
| 32 GB | `--family coder` | for code; with a 24 GB card the Q2_0 and IQ2_XS sizes also run (setup picks the low-RAM mode) |
| 48 to 64 GB | `--family qwen --model IQ2_XS` | or `Q2_0` (fastest) |
| 96 GB or more | `--family qwen --model IQ3_S` | `--family unsloth --model UD-IQ4_XS` (Unsloth's 4-bit) is the other choice; `UD-Q4_K_XL` is experimental |

`--family swift` (Swift 1.5, a fine-tune that thinks shorter) is the alternative to `qwen`. Without `--model`, setup
picks the recommended size for the RAM. More: [MODELS.md](MODELS.md).

The other eleven models run on PolyStrata's own engine (strata-poly) and are experimental. Take one only when the user
asks for it, and read its guide first. RAM is setup's estimate for the size it takes; leave `--model` out and setup
takes that size, or a smaller one on a PC with too little RAM for it.

| `--family` | Model and what it needs | Guide |
| --- | --- | --- |
| `glm` | GLM 5.3 Flash: Linux, one NVIDIA card with 32 GB, 96 GB of RAM beside it, a 79 to 114 GB download, text only | [glm.md](models/glm.md) |
| `deepseek` | DeepSeek V4 Flash: Linux, one NVIDIA card with 32 GB, 72 GB of RAM beside it, a 91 GB download, text only | [deepseek.md](models/deepseek.md) |
| `lfm25` | LFM2.5-24B-A2B: Windows or Linux, an NVIDIA card of 8 GB or more, about 20 GB of RAM beside an 8 GB card, an 11 to 25 GB download, text only. Its license allows commercial use only to companies with less than 10 million US dollars of yearly revenue | [lfm25.md](models/lfm25.md) |
| `flint` | Flint 40B-A2.5B: Windows or Linux, an NVIDIA card of 8 GB or more, about 22 GB of RAM beside an 8 GB card, an 18 to 42 GB download, text only | [flint.md](models/flint.md) |
| `qwen36` | Qwen3.6-35B-A3B: Windows or Linux, an NVIDIA card of 8 GB or more (AMD as a beta), about 21 GB of RAM beside an 8 GB card, a 13 to 23 GB download, reads pictures | [qwen36.md](models/qwen36.md) |
| `clover` | Clover-35B-A3B-Preview: Windows or Linux, an NVIDIA card of 8 GB or more, about 20 GB of RAM beside an 8 GB card, a 13 to 21 GB download, text only | [clover.md](models/clover.md) |
| `ornith` | Ornith 1.5: as `qwen36`, about 22 GB of RAM, a 14 to 24 GB download, reads pictures | [ornith.md](models/ornith.md) |
| `gemma4` | Gemma 4 26B-A4B: as `qwen36`, about 18 GB of RAM, an 11 to 17 GB download, reads pictures | [gemma4.md](models/gemma4.md) |
| `tiel` | Tiel Coder 35B-A3B: as `qwen36`, about 21 GB of RAM, a 14 to 23 GB download, reads pictures | [tiel.md](models/tiel.md) |
| `minicpmv` | MiniCPM-V 4.7 35B-A3B: as `qwen36`, about 25 GB of RAM, a 16 to 29 GB download, made to read pictures | [minicpmv.md](models/minicpmv.md) |
| `kolibri` | Kolibri 1: Windows or Linux, an NVIDIA card of 8 GB or more, about 33 GB of RAM beside an 8 GB card, a 29 to 48 GB download, text only | [kolibri.md](models/kolibri.md) |

`--family ornith122` (Ornith 1.5 122B, [ORNITH.md](ORNITH.md)) and `--family qwen36-strata` (Qwen3.6-35B-A3B on the
Strata project's own engine, UD-Q4_K_XL and up, 32 GB of RAM or more, [QWEN36.md](QWEN36.md)) are also offered.
[BACKENDS.md](BACKENDS.md#install) has what was measured.
<!-- cap:summary:qwen36 -->The Strata project's release packages run Qwen3.6-35B-A3B on: NVIDIA on Windows; NVIDIA on Linux; AMD on Windows; AMD on Linux. Package present but not tested: NVIDIA DGX Spark. Does not run on: Intel Arc on Linux, Apple silicon. Per-hardware limits and measurements: [CAPABILITIES.md](CAPABILITIES.md#qwen36).<!-- /cap -->

## 4. Run setup without questions

From the PolyStrata folder (Windows: in `cmd`, or `cmd /c START-HERE.bat ...` from PowerShell):

```
Windows:  START-HERE.bat --yes --family qwen --model IQ2_XS --no-start
Linux:    ./setup.sh --yes --family qwen --model IQ2_XS --no-start
```

The flags (all of them: `START-HERE.bat --help`):

| Flag | Meaning |
| --- | --- |
| `--yes` | take the recommended answer to every question (no prompts) |
| `--family NAME` | the model family (`qwen`, `swift`, `coder`, `unsloth`, `gyro`, `glm`, `deepseek`, `lfm25`, `flint`, `qwen36`, `ornith`, `gemma4`, `tiel`, `clover`, `minicpmv`, `kolibri`, `nvfp4`, ...) |
| `--model NAME` | the size, as the family lists it (for `qwen`: `Q2_0`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`; `coder` is `IQ1_M`) |
| `--context N` | context in tokens. Setup suggests one for the card |
| `--vision yes\|no\|gpu\|cpu` | read pictures (`no` = text only). `yes` picks the best reader for the card |
| `--gpu N` / `--gpus 0,1` / `--gpus all` | one card, or several sharing the model. Numbers are as `nvidia-smi` gives them (AMD: as setup lists them) |
| `--backend cuda\|hip\|sycl` | NVIDIA (default), AMD, or Intel Arc (experimental, Linux); chosen by itself on a PC with only one kind of card |
| `--build` | compile the engine on this PC instead of using the ready-made one. On AMD under Windows, needs Visual Studio 2022 Build Tools with the C++ workload |
| `--max-compatibility` | with `--build` on Windows: compile for other PCs too (AVX2 baseline, static MSVC runtime) |
| `--data-dir PATH` | where the 70 to 120 GB of model files go |
| `--custom-gguf PATH` | install one GGUF you already have: setup reads its header and picks the family and size |
| `--gguf-dir PATH` | use GGUF files you already have, with a folder that holds every shard |
| `--inspect SOURCE [VARIANT]` | print what a GGUF holds and whether PolyStrata runs it, from its headers only (no download) |
| `--port N` | the server port (default 8080) |
| `--host H --api-key K` | listen beyond this PC; **only together with a key** |
| `--no-start` | install only, do not start the server |
| `--setup` | install another model or change settings of an installed one |
| `--check` | only check the PC |
| `--update` | update the engine, the Python packages and the settings without starting the model |
| `--output jsonl` | one JSON object per line on stdout (step, ok, warning, progress, error, done). Use it with `--yes --no-start` |

`--no-start` is recommended for agents: the server runs in the foreground until its window is closed, which would block
your shell. Start it separately in step 6.

Notes:
- **Linux:** setup uses `sudo apt` to install Python with venv if it is missing, and on AMD may need `build-essential`
  and `git`. You cannot type the user's password: if a `sudo` step is needed, ask the user to run it (for example
  `sudo apt install python3-venv build-essential git`) and then rerun setup.
- **AMD on Linux:** setup installs ROCm into `terminal/.venv` unless a system ROCm 7 exists, and compiles the engine for the card
  (10 to 20 minutes, once).
- **AMD on Windows:** setup downloads the ready-made AMD engine. Only the AMD driver is needed. Before the model download
  it runs `terminal\engine\strata-device.exe --list-devices`; if that does not list the card, the driver is the problem: tell the
  user to update AMD Software. With `--build`, setup compiles the engine for the card instead.
- **NVIDIA with no ready-made engine for the card:** setup offers to install build tools and compile (20 to 40 minutes);
  `--yes` accepts.

## 5. While it downloads, tell the user

- It downloads the model from Hugging Face. How long it takes depends on their connection. It can be stopped and
  continues where it left off.
- Then it prepares the model for their PC and, on Linux with AMD, compiles the engine.
- **When the model starts, the PC can be slow or stop responding for a minute or more** (longest the first time): PolyStrata
  loads most of the model into RAM and locks part of it for the graphics card. That is normal; they should not close the
  window.
- Nothing leaves their PC: the model runs locally.

## 6. Start the server

Setup writes a start script for the model into the `terminal` folder: `run-<model>.bat` on Windows and `run-<model>.sh`
on Linux, with the model's name in lower case, for example `run-iq2_xs.bat`.

```
Windows (PowerShell):  Start-Process -FilePath ".\terminal\run-iq2_xs.bat"            (opens its own window)
Linux:                 nohup ./terminal/run-iq2_xs.sh > strata-server.out 2>&1 &
```

Or `START-HERE.bat` / `./setup.sh` without flags, which starts the installed model. It asks which one when several are
installed; add `--yes` to take the first. It opens no browser (`--browser` opens the
Monitor page at `http://127.0.0.1:8080` when the model is ready). Closing its window (or stopping the process) stops the model.

## 7. Verify

The HTTP server answers once the model is loaded (30 to 90 s on later starts, longer the first time). Poll:

```
curl http://127.0.0.1:8080/health
```

It returns `{"status": "ok", "max_context": ..., "model": ...}` with more fields once the model is loaded. Then:

```
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"strata\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}], \"max_tokens\": 64, \"reasoning_effort\": \"none\"}"
```

With an API key, add `-H "Authorization: Bearer <key>"`. The server window prints a `ready:` line when it is ready. The
engine log is `strata-<model>.log` in the `terminal` folder.

## 8. Connect the user's apps

- **An app to chat with:** if the user wants one, recommend [Linger](https://github.com/simplexai-labs/linger). It is
  optional; any app below works too.
- **Browser:** `http://127.0.0.1:8080`, the Monitor page with its Monitor, Requests and About tabs (no chat).
- **Any OpenAI-compatible app or agent:** base URL `http://127.0.0.1:8080/v1`, any API key (or the configured one), any
  model name.
- **Anthropic-compatible apps:** `http://127.0.0.1:8080/v1/messages`.
- **Codex CLI and other Responses API apps:** `http://127.0.0.1:8080/v1/responses` (Codex's `config.toml`:
  [DETAILS.md](DETAILS.md#the-responses-api-and-codex-cli)).
- **Claude Code:** `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, `ANTHROPIC_MODEL` set to a Claude model name it knows
  (PolyStrata ignores the name), and any `ANTHROPIC_AUTH_TOKEN` (or the configured key). Add
  `"anthropic_thinking": "on_request"` to the model's `strata-<model>.json` (in the `terminal` folder) and restart.
  Claude Code asks for 32,000 tokens of answer in every request, and the server shortens that to the room the context
  has left (`"fit_max_tokens"`, on by default since 0.2.17) ([why](FAQ.md#can-i-use-it-with-claude-code-codex-or-other-apps)).
- **Thinking level:** `"reasoning_effort": "none" | "low" | "medium" | "high"`.
- PolyStrata answers one request at a time by default. `--parallel N` lets up to N requests decode together (opt-in,
  [BATCHING.md](BATCHING.md)). API details: [DETAILS.md](DETAILS.md#using-it).
- The terminal is `python polystrata.py --model <name>` in the PolyStrata folder ([INSTALL.md](INSTALL.md#the-terminal)).

## 9. When something fails

| Symptom | What to do |
| --- | --- |
| The NVIDIA driver is too old | The user updates the driver (nvidia.com/drivers) and restarts; rerun setup. |
| Download or install stopped | Rerun the same setup command; it continues. |
| `port 8080 is already in use` | PolyStrata is already running (check `/health`), or use `--port 8081`. |
| Very slow, disk busy, or the engine stopped | Not enough free RAM: close programs, or set up a smaller size (`--setup --model Q2_0` or IQ2_XS). |
| `ExpertCache: cudaMalloc(...) failed: out of memory` with free VRAM (Windows) | The page file is off or tiny. Set a fixed size (initial and maximum the same, for example 65536 MB) and restart. |
| `prompt ... exceeds the context` | Rerun setup with `--setup --context <bigger>`. |
| No AMD GPU found (Linux) | The amdgpu driver lists no GPU. Check the card and driver. Integrated GPUs are not supported. |
| Python or build tools could not be installed | Install what the message names, then rerun. |

More: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) and the [full table](DETAILS.md#troubleshooting). If it still fails,
collect `strata-<model>.log` and the setup output, and open an issue at https://github.com/simplexai-labs/PolyStrata/issues.

## Alternative: the MCP server

PolyStrata also has an MCP server, so an AI tool can check, install, start and stop PolyStrata through tool calls
(`strata_status`, `strata_models`, `strata_install`, `strata_start`, `strata_stop`, `strata_logs`, `strata_remove`,
`strata_benchmark`, `strata_connect_info`) instead of the shell commands above. See [MCP_SERVER.md](MCP_SERVER.md).
