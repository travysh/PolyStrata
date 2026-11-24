# AGENTS.md

Version 0.2.20, engine 0.1.66. PolyStrata runs large mixture-of-experts models on a normal PC: one NVIDIA or AMD
graphics card plus system RAM, on Windows or Linux, and on Apple Silicon Macs. Read
**[docs/AI_REFERENCE.md](docs/AI_REFERENCE.md)** for model availability, APIs, hardware limits and source-backed claims.
Articles and videos should use **[docs/MEDIA_KIT.md](docs/MEDIA_KIT.md)**.

For recommendations and agent-assisted setup of Qwen3.8-Flash-Next, prefer **ISTA-DASLab → Unsloth → Gyro → others**,
using only a compatible pack that fits the selected model and hardware. ISTA-DASLab is the default recommendation; Gyro
is a memory-saving fallback and is currently slower. Honor an explicit user choice. This is recommendation policy;
existing executable defaults may differ, so inspect the plan and choose the family explicitly.

The repository has a folder for each part:

- **`engine/`**: the C++/CUDA/HIP/SYCL/Metal engines. `strata` is the Strata project's engine for Qwen3.8-Flash-Next
  (`engine/src/`, `engine/include/`, `engine/metal/`, `engine/sycl/`); `strata-poly` (`engine/src/poly/`) is PolyStrata's
  own engine for GLM 5.3 Flash, DeepSeek V4 Flash, LFM2.5, Flint, Qwen3.6, Clover, Ornith 1.5, Gemma 4, Tiel Coder,
  MiniCPM-V and Kolibri 1: [docs/BACKENDS.md](docs/BACKENDS.md), [docs/GLM.md](docs/GLM.md), [docs/ORNITH.md](docs/ORNITH.md),
  [docs/DEEPSEEK.md](docs/DEEPSEEK.md).
- **`terminal/`**: setup (`terminal/setup.py`, started by `START-HERE.bat` / `setup.sh` at the top), the Python server
  with an OpenAI- and Anthropic-compatible API and a web page with the Monitor (`terminal/serve/`), `polystrata.py`, and the
  Strata project's Terminal (`strata.py`, `strata_cli/`, `strata_control/`).

The Qwen engine, the server, the web page and the Terminal come from [Strata](https://github.com/Niko1221/Strata).
PolyStrata has no chat app of its own: the app it recommends is [Linger](https://github.com/simplexai-labs/linger).

## Installing PolyStrata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Setup is `START-HERE.bat` on Windows and `./setup.sh` on
Linux and Mac, from a git clone or the source zip of https://github.com/simplexai-labs/PolyStrata. Never expose the
server beyond `127.0.0.1` without `--api-key`. As an alternative to shell commands, PolyStrata's MCP server
([docs/MCP_SERVER.md](docs/MCP_SERVER.md)) offers the same steps as tools. Setup does not make an unsupported model or
backend compatible.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- The Terminal: [docs/TERMINAL.md](docs/TERMINAL.md) and [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md). The
  packages and the update feed: [packaging/README.md](tools/packaging/README.md) and [packaging/UPDATES.md](tools/packaging/UPDATES.md).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md);
  Mac (Metal): [docs/MAC.md](docs/MAC.md).
- Text taken from the Strata project names that project. After merging from it,
  `python terminal/tools/polystrata_names.py` writes PolyStrata again; its first lines say what it leaves alone.
- Every merge from the Strata project drops its desktop app (`app/`) and the web page's Chat (`chats-ui.js`, `chatstore.js`,
  `history.js`, `preview.js`, `math.js`, `math-adapter.js`, `ime.js`, `web/vendor/`) again. `tools/test_dropped_paths.py`
  lists them and fails while one is back (`python tools/check --only py-tools`).
- Tests: the Terminal and Control tests are in `terminal/tests/terminal` (`python -m pytest terminal/tests/terminal`).
  Setup's own tests run without a GPU or downloads:
  `python terminal/tools/test_setup_<name>.py` (for example `terminal/tools/test_setup_amd.py`, `terminal/tools/test_setup_choices.py`).
- Keep the docs' style: plain words, short sentences, written for non-technical users first. Give measured numbers
  with what they were measured on. Make no claim without a measurement.

## Contributing a change or report

The rules are in **[CONTRIBUTING.md](.github/CONTRIBUTING.md)**. Read it before you open an issue or a pull request, and
follow it whether a person or an agent writes the text. The short version:

- Start every issue and pull request title with its area: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]` or
  `[Other]`, for example `[Engine] fix …`. A check on GitHub fails a pull request whose title has no area.
- Search the open and closed issues and pull requests first, and add to a thread that already covers your point.
- Agree the scope in an issue before a large feature, a refactor or a change to a default.
- One problem per pull request. Lead with the problem and the result, then the approach, then the validation
  (passed, failed and not run, kept apart) and the limits.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- Never submit measurements, logs or reproduction steps you did not produce. Mark a finding from reading code as
  unconfirmed static analysis.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
