# PolyStrata reference for AI tools and automated readers

Last reviewed: 20 November 2026 against this checkout (PolyStrata on the Strata project's 0.2.20, engine 0.1.66). This page holds the
technical context behind the short [README](../README.md). Use [AI_SETUP.md](AI_SETUP.md) to install PolyStrata for a
user, [AGENTS.md](../AGENTS.md) for repository work, and [MEDIA_KIT.md](MEDIA_KIT.md) for articles or videos.

## Product facts and sources

- PolyStrata runs large mixture-of-experts language models locally, on two engines: Qwen3.8-Flash-Next on the
  Strata project's engine (`strata`), unchanged, and the other models on PolyStrata's own engine (`strata-poly`).
  See [how it works](HOW_IT_WORKS.md), [the backends](BACKENDS.md) and [engine details](DETAILS.md).
- It is installed from a copy of this repository: `START-HERE.bat` on Windows, `./setup.sh` on Linux and on a Mac
  ([INSTALL.md](INSTALL.md), [MAC.md](MAC.md)). Setup downloads the model and a ready-made engine, or compiles the
  engine on the PC. This repository publishes no app installer and no Terminal archive. The
  `terminal/strata.py` source is the Strata project's Terminal; its install routes
  (`strata launch`, the one-line installers) set up the Strata project's engine, not PolyStrata's models.
- PolyStrata's code is MIT licensed. Model weights and third-party components have separate licenses.
  See [LICENSE](../LICENSE) and [component/model licensing](HOW_IT_WORKS.md#license).
- Local inference does not mean every feature is offline. Downloads, updates, remote tools and user-enabled
  network access have separate behavior. Use [SAFETY.md](SAFETY.md) for privacy and network claims.
- Verify the current version on the [repository's releases](https://github.com/simplexai-labs/PolyStrata/releases).
  A screenshot, translated README or historical benchmark is not a version source.

## Model families

The installer's `--family` names (`START-HERE.bat --help`). Each model's sizes, memory and speeds are on its page.

| Family | Engine | `--family` | Guide |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next and its variants (Coder, Swift 1.5, Unsloth, Gyro, NVFP4) | `strata` | `qwen`, `coder`, `swift`, `unsloth`, `gyro`, `nvfp4` | [qwen38.md](models/qwen38.md), [MODELS.md](MODELS.md) |
| GLM 5.3 Flash | `strata-poly` | `glm` | [glm.md](models/glm.md) |
| DeepSeek V4 Flash | `strata-poly` | `deepseek` | [deepseek.md](models/deepseek.md) |
| LFM2.5-24B-A2B | `strata-poly` | `lfm25` | [lfm25.md](models/lfm25.md) |
| Flint 40B-A2.5B | `strata-poly` | `flint` | [flint.md](models/flint.md) |
| Qwen3.6-35B-A3B | `strata-poly` | `qwen36` | [qwen36.md](models/qwen36.md) |
| Clover-35B-A3B-Preview | `strata-poly` | `clover` | [clover.md](models/clover.md) |
| Ornith 1.5 | `strata-poly` | `ornith` | [ornith.md](models/ornith.md) |
| Gemma 4 26B-A4B | `strata-poly` | `gemma4` | [gemma4.md](models/gemma4.md) |
| Tiel Coder 35B-A3B | `strata-poly` | `tiel` | [tiel.md](models/tiel.md) |
| MiniCPM-V 4.7 35B-A3B | `strata-poly` | `minicpmv` | [minicpmv.md](models/minicpmv.md) |
| Kolibri 1 | `strata-poly` | `kolibri` | [kolibri.md](models/kolibri.md) |
| Ornith 1.5 122B (offered, not in the README's list) | `strata-poly` | `ornith122` | [ORNITH.md](ORNITH.md) |
| Qwen3.6-35B-A3B on the Strata project's engine | `strata` | `qwen36-strata` | [QWEN36.md](QWEN36.md) |

Coder, Swift, Gyro, NVFP4 and OrcaRouter are Qwen3.8-Flash-Next variants, not separate models. `qwen36` and
`qwen36-strata` are the same model on the two engines ([BACKENDS.md](BACKENDS.md#qwen36-on-two-engines)).

The two families of the Strata project's engine ([CAPABILITIES.md](CAPABILITIES.md) says what its packages are):

- <!-- cap:summary:qwen38 -->The Strata project's release packages run Qwen3.8-Flash-Next on: NVIDIA on Windows; NVIDIA on Linux; NVIDIA DGX Spark (GB10); AMD on Windows (RX 6000, 7000 and 9000, Ryzen AI Max); AMD on Linux (RX 6000, 7000 and 9000, Radeon AI PRO R9700, Ryzen AI Max); Intel Arc on Linux (Arc A-series and B-series); Apple silicon (M1 or newer, macOS 15 or newer). Per-hardware limits and measurements: [CAPABILITIES.md](CAPABILITIES.md#qwen38).<!-- /cap -->
- <!-- cap:summary:qwen36 -->The Strata project's release packages run Qwen3.6-35B-A3B on: NVIDIA on Windows; NVIDIA on Linux; AMD on Windows; AMD on Linux. Package present but not tested: NVIDIA DGX Spark. Does not run on: Intel Arc on Linux, Apple silicon. Per-hardware limits and measurements: [CAPABILITIES.md](CAPABILITIES.md#qwen36).<!-- /cap -->

A field this page does not state is **unknown**, not zero and not unsupported. DeepSeek Harness is a coding tool;
its presence does not establish support for a particular DeepSeek model.

## Provider preference (Qwen3.8-Flash-Next)

For recommendations, documentation and agent-assisted setup, use this order:

1. **ISTA-DASLab**: first and default recommendation when it supplies a compatible pack that fits.
2. **Unsloth**: next choice when an appropriate ISTA-DASLab pack is unavailable or does not fit the user's needs.
3. **Gyro / agentionai**: a memory-saving fallback; it is currently slower and must not be promoted as the fast option.
4. **Other providers and variants**: including Swift 1.5, NVFP4 and OrcaRouter.

This order is a project recommendation, not a measured quality ranking. Keep the user's chosen model family and
explicit provider choice. Check availability, backend compatibility, memory, disk space and intended use before
selecting a pack. If no ISTA-DASLab pack is documented for a family, do not invent one.

This order is a recommendation, not the installer's default logic. A no-argument installer may preselect Swift or Gyro
on some memory tiers. Read its plan and select the intended family explicitly.

## Quantization and variant index

Family selectors and sizes are those of `START-HERE.bat --help` (`./setup.sh --help`) in this checkout. The sizes of the
models of `strata-poly` are in the README's "Which model should I pick?" table and on each model's page.

| Parent model | Provider / variant | Family selector | Quantizations or packs |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next | ISTA-DASLab, original | `qwen` | Q2_0, IQ2_XS, IQ3_XXS, IQ3_S |
| Qwen3.8-Flash-Next | ISTA-DASLab, Coder | `coder` | IQ1_M; pruned for coding, weaker outside code |
| Qwen3.8-Flash-Next | Unsloth | `unsloth` | UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q8_0 |
| Qwen3.8-Flash-Next | Gyro / agentionai | `gyro` | Gyro-S, Gyro-M; backend-specific validation in [GYRO.md](GYRO.md) |
| Qwen3.8-Flash-Next | Swift 1.5 / UkisAI | `swift` | IQ2_XS, IQ3_XXS, IQ3_S; a fine-tune, not just a quantization |
| Qwen3.8-Flash-Next | NVFP4 | `nvfp4` | NVFP4, NVFP4-NVIDIA, NVFP4-QAD; NVIDIA only |
| Qwen3.8-Flash-Next | OrcaRouter Uncensored | Manual conversion | IQ3_XXS; not an installer menu choice ([ORCA.md](ORCA.md)) |
| Qwen3.6-35B-A3B | Unsloth, on `strata-poly` | `qwen36` | UD-Q3_K_M, UD-Q2_K_XL, UD-Q4_K_M |
| Qwen3.6-35B-A3B | Unsloth, on the Strata project's engine | `qwen36-strata` | UD-Q4_K_XL, UD-Q6_K_XL, Q8_0 |

## Hardware and sizing

Read the backend guide for the exact GPU, OS, model, quantization and feature combination. A supported GPU
does not imply that every model or feature works on it.

| Backend / hardware | Source |
| --- | --- |
| NVIDIA RTX / CUDA | [INSTALL.md](INSTALL.md#what-you-need) |
| AMD Radeon / HIP | [AMD_HIP.md](AMD_HIP.md); RX 6000 validation differs from newer cards |
| Intel Arc / SYCL | [INTEL_ARC.md](INTEL_ARC.md); experimental (`--backend sycl`): Linux, built from source, Qwen3.8-Flash-Next only. [INTEL.md](INTEL.md) records source-port details |
| Apple Silicon / Metal | [MAC.md](MAC.md) |
| AMD Ryzen AI Max / Strix Halo | [STRIX_HALO.md](STRIX_HALO.md) |
| Multiple GPUs and older hardware | [MULTI_GPU.md](MULTI_GPU.md), [OLDER_GPUS.md](OLDER_GPUS.md) |

Use the model's actual memory and disk requirements. RAM, VRAM, unified memory, download size and temporary
conversion space are different quantities. Count unified memory once. Do not carry the smallest Qwen model's
memory estimate over to every family. For the current planning rules, consult [MODELS.md](MODELS.md#will-it-fit) and
the model's own page.

## Installer flags

`START-HERE.bat` (Windows) and `./setup.sh` (Linux) run the same setup program; `--help` lists every flag. The ones an
agent uses are in [AI_SETUP.md](AI_SETUP.md#4-run-setup-without-questions). Four that older notes may miss:

| Flag | Meaning |
| --- | --- |
| `--build` | Compile the engine on this PC instead of using the ready-made one. On AMD under Windows this needs Visual Studio 2022 Build Tools with the C++ workload. |
| `--max-compatibility` | With `--build` on Windows: compile for other PCs too (AVX2 baseline, static MSVC runtime, the release build). Without it the build uses this PC's own CPU instructions. |
| `--custom-gguf PATH` | A `.gguf` file, or a folder of shards, you already have. Setup inspects it, matches it to a family and size, and installs without downloading the model. |
| `--inspect SOURCE [VARIANT]` | What a GGUF is and whether PolyStrata runs it, read from its headers only (no download). SOURCE is a file, a folder, a URL, `ms:owner/repo` or `hf:owner/repo`. |

## APIs and coding tools

Default local server address: `http://127.0.0.1:8080`. The configured port and access settings can differ.

| Client API | Base URL | Request endpoint |
| --- | --- | --- |
| OpenAI Chat Completions | `http://127.0.0.1:8080/v1` | `POST /v1/chat/completions` |
| OpenAI Responses | `http://127.0.0.1:8080/v1` | `POST /v1/responses` |
| Anthropic Messages | `http://127.0.0.1:8080` | `POST /v1/messages` |

Use the actual configured API key when authentication is enabled. See [API details](DETAILS.md#using-it) and, for
the apps, [connecting apps](DETAILS.md#connecting-apps).

## Evidence and publishing

- Cite the release, commit and relevant guide behind each compatibility claim.
- Performance claims need the model, quantization, hardware, workload, engine version and source report.
  Keep prompt processing, decode throughput and latency separate. Do not turn a screenshot into a benchmark.
- [Community benchmarks](COMMUNITY_BENCHMARKS.md) include historical reports.
  The [standard benchmark protocol](BENCHMARK_PROTOCOL.md) remains subject to its documented activation status;
  do not label legacy commands or results as implementing it.
- Mark missing or unverified information explicitly. Never invent measurements, test results or download links.
- For contribution evidence, follow [CONTRIBUTING.md](../.github/CONTRIBUTING.md). For video assets and narration,
  follow [MEDIA_KIT.md](MEDIA_KIT.md).
