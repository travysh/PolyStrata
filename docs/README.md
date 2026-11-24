# PolyStrata guides

**New here? Start with [Installing PolyStrata](INSTALL.md).** Download PolyStrata and run `START-HERE.bat` (Windows) or `./setup.sh` (Linux, Mac): it installs the model and the engine and starts it. The models themselves are in the [README](../README.md#which-model-should-i-pick).

## Use PolyStrata

| I want to… | Guide |
| --- | --- |
| Install or update | [Installing PolyStrata](INSTALL.md) |
| Choose a model and check memory | [The README](../README.md#which-model-should-i-pick), each model's page and, for Qwen3.8-Flash-Next, [MODELS.md](MODELS.md) |
| See what each model needs | [The README](../README.md#what-you-need) and [BACKENDS.md](BACKENDS.md#install) |
| Chat or benchmark from the command line | [The terminal](INSTALL.md#the-terminal) |
| Connect a coding tool | [Connecting apps](DETAILS.md#connecting-apps) |
| Fix an install or startup problem | [Troubleshooting](TROUBLESHOOTING.md) |
| Check privacy, downloads and network access | [Safety and privacy](SAFETY.md) |

## Hardware

Use the guide for your GPU and operating system. Model and feature support can differ between backends.

| Hardware | Guide |
| --- | --- |
| NVIDIA RTX 20 / 30 / 40 / 50 | [Install](INSTALL.md#what-you-need) |
| AMD Radeon (RX and AI PRO R9700) | [AMD / HIP](AMD_HIP.md) |
| AMD Ryzen AI Max / Strix Halo | [Strix Halo](STRIX_HALO.md), [on Windows](STRIX_HALO_WINDOWS.md) |
| Intel Arc on Linux | [Intel Arc](INTEL_ARC.md) |
| Apple Silicon | [Mac / Metal](MAC.md) |
| More than one GPU | [Multiple GPUs](MULTI_GPU.md) |
| Two cards: one reads the prompt, one makes the answer (opt-in) | [Split serving](SPLIT_SERVE.md) |
| Older GPUs | [Experimental hardware](OLDER_GPUS.md) |

## Developers and contributors

- [Build or run from a source checkout](SOURCE_INSTALL.md)
- [Docker](DOCKER.md)
- [Contribution rules](../.github/CONTRIBUTING.md)
- [Community benchmarks](COMMUNITY_BENCHMARKS.md) and the [standard benchmark protocol](BENCHMARK_PROTOCOL.md)
- [How the engine works](HOW_IT_WORKS.md), [technical reference](DETAILS.md) and [Terminal reference](TERMINAL.md)
- [KVA: approximate prompt processing for the late layers (experimental, opt-in)](KVA.md)
- [Packaging](../tools/packaging/README.md)

## AI tools and media

- [Install for a user](AI_SETUP.md)
- [Verified facts, model availability and API reference](AI_REFERENCE.md)
- [Articles and videos](MEDIA_KIT.md)
- [Repository instructions for agents](../AGENTS.md)

## Which instructions apply?

`START-HERE.bat`, `UPDATE.bat` and `setup.sh` are PolyStrata's setup; raw engine options in a model's run config are a
second interface. `strata launch` and the other `strata` commands belong to the Strata project's Terminal, which this
repository carries in `terminal/` ([TERMINAL_LAUNCH.md](TERMINAL_LAUNCH.md)); it sets up Qwen3.8-Flash-Next only. Their
flags are not interchangeable: the Terminal uses `--gpu` and `--vision on|off`, setup uses `--gpus` and `--vision yes|no`.

Release notes, design proposals and benchmark reports describe their named versions and machines. They remain
available as historical evidence; use the guides above for a new installation. A published result on one GPU
does not establish support or speed on every GPU.

## Writing a guide page

- Installation steps live in [INSTALL.md](INSTALL.md), source steps in [SOURCE_INSTALL.md](SOURCE_INSTALL.md). The public READMEs stay short.
- Check command examples against the parser and filenames against the release assets.
- A guide describes the current release. Version history, fixed bugs and superseded settings go in a closing
  **History / older measurements** section, in `docs/bench-results`, or in the release notes. Do not mix them into instructions.

Every guide page has this shape:

| Part | Content |
| --- | --- |
| Title | `# Name of the task or topic` |
| Purpose | One to three lines: what the page is for and who it is for |
| Quick start or Steps | The shortest working path, commands in code blocks |
| Details | `##` sections; option and flag lists as tables; short paragraphs |
| History / older measurements | Optional, last. Dated numbers only |

Label every measurement the same way, directly above the number or table:

> **Measured on:** hardware (GPU, CPU, RAM), model or set, PolyStrata version, workload.
