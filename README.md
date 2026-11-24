<h1 align="center">PolyStrata</h1>

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Türkçe](README.tr.md)

<p align="center"><b>Run large AI models on your own PC</b><br>
Qwen3.8-Flash-Next, Hy4-Flash, GLM 5.3 Flash, DeepSeek V4 Flash, LFM2.5, Flint, Qwen3.6, Clover, Ornith 1.5, Gemma 4, Tiel Coder, MiniCPM-V and Kolibri 1 · free and open source</p>

<p align="center"><a href="https://github.com/simplexai-labs/linger"><img height="28" alt="Used by Linger" src="docs/media/used-by-linger.svg"></a></p>

<p align="center"><img src="docs/media/speed.svg" width="820" alt="Tokens per second while writing an answer: Qwen3.6-35B-A3B writes 106 on a laptop with an 8 GB card and 536 on an RTX 5090"></p>

<p align="center"><img src="docs/media/claude-code-demo.gif" width="820" alt="Claude Code in a terminal beside PolyStrata's API Monitor"><br>
<sub>Claude Code on Qwen3.6-35B-A3B UD-Q3_K_M, on the laptop with the 8 GB card, in real time: it adds a <code>--json</code> option to a small program, writes a test for it and runs the tests. The server had read Claude Code's system prompt when it started (<code>"warm_start": true</code>).</sub></p>

<p align="center"><img src="docs/media/linger-inkling-3d.gif" width="820" alt="Linger on PolyStrata writing a 3D village"><br>
<sub><a href="https://github.com/simplexai-labs/linger">Linger</a> on PolyStrata: Inkling 2 Mini UD-Q3_K_XL on the laptop with the 8 GB card, about 33 tokens/s, writes a low-poly village in one three.js file, and Linger opens it.</sub></p>

PolyStrata runs large AI models on a normal PC, models that usually need a server. They chat, write code and work
with your apps and coding agents. Nothing leaves your PC.

It runs fourteen models, on two engines:

- **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** on an NVIDIA or AMD graphics card with
  12 GB or more, on Windows or Linux. It also reads pictures. Its engine is the Strata project's, unchanged.

The other thirteen run on an engine written for PolyStrata, with one NVIDIA card or [two](docs/MULTI_GPU.md#the-strata-poly-models): on Linux, and all but GLM 5.3 Flash and
DeepSeek V4 Flash on Windows too. They are experimental: they have run on one PC with a 32 GB card, nine of the smaller ones on a laptop
with an 8 GB card as well, and Groot on a PC with an 8 GB RTX 5060.

- **[GLM 5.3 Flash](docs/models/glm.md)**, the largest: a file of 79 to 114 GB, most of which stays in RAM.
- **[DeepSeek V4 Flash](docs/models/deepseek.md)**, a file of 91 GB, most of which stays in RAM too. Its memory of a
  conversation is small: a 128K context takes 0.7 GB more VRAM than 32K.
- **[LFM2.5-24B-A2B](docs/models/lfm25.md)** by Liquid AI, a file of 11 to 25 GB. Most of its layers have no attention, so a
  long conversation takes little VRAM: a 128K context takes 2 GB more than 32K. Its license allows commercial use
  only to companies with less than 10 million dollars of yearly revenue.
- **[Flint 40B-A2.5B](docs/models/flint.md)** by Recursive, made for coding agents: a file of 18 to 42 GB. Most of its layers
  look back only 513 tokens, so a long conversation takes little VRAM: a 128K context takes 2.6 GB more than 32K.
- **[Qwen3.6-35B-A3B](docs/models/qwen36.md)**, a file of 13 to 23 GB. It also reads pictures.
- **[Groot 19B-A3B 261023](docs/models/groot.md)**, experimental setup integration for three of Bartowski's quantizations,
  Q4_K_M, Q6_K and Q8_0 (12.5 to 20.2 GB), with MTP and an optional image projector. Q4_K_M text has run on an
  8 GB RTX 5060 and on an RTX 5090 under Linux; images, Q6_K and Q8_0 are still pending.
- **[Inkling 2 Mini](docs/models/inkling.md)** by Thinking Machines, experimental: a file of 9 to 18 GB. It has run
  on Linux with a 32 GB NVIDIA card and on Windows with an 8 GB laptop card. It reads pictures too, one request at a time.
- **[Clover-35B-A3B-Preview](docs/models/clover.md)** by DiffuSpace, Qwen3.6-35B-A3B trained further to also write a
  block of tokens at once: a file of 13 to 21 GB. It reads text only.
- **[Ornith 1.5](docs/models/ornith.md)**, which has Qwen3.6-35B-A3B's architecture and other weights: a file of 14 to 24 GB. It
  also reads pictures.
- **[Gemma 4 26B-A4B](docs/models/gemma4.md)**, a file of 11 to 17 GB. It also reads pictures.
- **[Tiel Coder 35B-A3B](docs/models/tiel.md)**, Ornith 1.5 quantized for coding agents by peculiar-ragdoll: a file of
  14 to 23 GB. It also reads pictures.
- **[MiniCPM-V 4.7 35B-A3B](docs/models/minicpmv.md)** by OpenBMB, which has Qwen3.6-35B-A3B's architecture and weights of its
  own, made to read pictures: a file of 16 to 29 GB.
- **[Kolibri 1](docs/models/kolibri.md)** by Aleph Alpha, which writes German as well as English: a file of 29 to 48 GB,
  most of which stays in RAM beside a small card.
- **[Hy4-Flash](https://github.com/Niko1221/Strata/releases/tag/v0.2.18)** by Tencent, 220B-A12B, on the Strata project's own engine (`strata-hy4`): a file of 130 to
  181 GB, from 64 GB of RAM with an 8 GB card. It also reads pictures.

The server can change between installed models while it runs ([how](docs/BACKENDS.md#changing-the-model)).

PolyStrata was created by [VecSzn](https://github.com/VecSzn).
PolyStrata is built on [Strata](https://github.com/Niko1221/Strata) by Niko1221.
Why it is a separate repo, AMD and smaller cards, fine-tunes and other questions: [FAQ](docs/FAQ.md).

## How fast is it?

A token is about ¾ of a word.

- **Writes answers:** how fast the reply appears in a short chat. 60 tokens per second is faster than you can read.
- **Reads your prompt:** how fast it takes in what you send (a document, code or a chat history).

The table gives the size the installer takes for each model, in tokens per second. Every size's numbers, the comparison with llama.cpp and how they were measured are on each model's page.

| Model | Laptop with an 8 GB card<br>writes / reads | RTX 5090<br>writes / reads | Every size |
| --- | ---: | ---: | --- |
| LFM2.5-24B-A2B | 86 / 4,035 | 492 / 25,275 | [LFM2.5](docs/models/lfm25.md) |
| Flint 40B-A2.5B | 64 / 2,643 | 261 / 15,033 | [Flint](docs/models/flint.md) |
| Qwen3.6-35B-A3B | 106 / 2,483 | 536 / 15,319 | [Qwen3.6](docs/models/qwen36.md) |
| Clover-35B-A3B-Preview | 83 / 2,242 | 308 / 13,918 | [Clover](docs/models/clover.md) |
| Ornith 1.5 | 87 / 2,362 | 366 / 14,438 | [Ornith](docs/models/ornith.md) |
| Gemma 4 26B-A4B | 92 / 2,187 | 370 / 11,478 | [Gemma 4](docs/models/gemma4.md) |
| Tiel Coder 35B-A3B | 89 / 2,517 | 440 / 15,109 | [Tiel Coder](docs/models/tiel.md) |
| MiniCPM-V 4.7 35B-A3B | 79 / 2,263 | 331 / 14,615 | [MiniCPM-V](docs/models/minicpmv.md) |
| Kolibri 1 | 66 / 851 | 207 / 6,133 | [Kolibri 1](docs/models/kolibri.md) |
| GLM 5.3 Flash | does not run: needs a 32 GB card | 52 / 863 | [GLM](docs/models/glm.md) |
| DeepSeek V4 Flash | does not run: needs a 32 GB card | 54 / 1,926 | [DeepSeek](docs/models/deepseek.md) |
| Inkling 2 Mini | 53 / 892 | 196 / 6,706 | [Inkling 2 Mini](docs/models/inkling.md) |
| Groot 19B-A3B 261023 | 97 / 2,785 | 458 / 15,634 | [Groot](docs/models/groot.md) |
| Hy4-Flash | does not run: needs 64 GB of RAM | 90-111 / 1,694 | [Hy4-Flash](https://github.com/Niko1221/Strata/releases/tag/v0.2.18) |

Laptop: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM. Workstation: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 90 to 120 GB RAM. Flint is UD-Q3_K_XL on the laptop, the size the installer takes with 32 GB of RAM, and UD-Q4_K_XL on the workstation. Kolibri 1 is Q2_K on both; the workstation's installer takes Q4_K_M, which was not measured.

Qwen3.8-Flash-Next and Hy4-Flash run on the Strata project's engines and were measured by that project. Qwen3.8-Flash-Next on a PC with an RTX 5070 (12 GB) and 64 GB of RAM: IQ2_XS writes at 79 and reads a 32K prompt at 2,090 ([every size](docs/models/qwen38.md)). Hy4-Flash on an RTX 5090: 90 to 111, and a 128K prompt read at 1,694 ([its release](https://github.com/Niko1221/Strata/releases/tag/v0.2.18)).

The thirteen models of PolyStrata's own engine were measured with its benchmark, which you can run on your PC:
`python polystrata.py benchmark --model <name> --prompt-tokens 24000`
([what it sends and prints](docs/BACKENDS.md#benchmark)).
What others measured on their own cards, and how to compare with llama.cpp on yours:
[community results](docs/COMMUNITY_BENCHMARKS.md#the-models-polystrata-adds).
On the laptop PolyStrata read prompts 2.7 to 3.5 times as fast as llama.cpp b11540 and wrote 1.5 to 1.8 times
as fast, with Qwen3.6 and with Tiel Coder ([the two runs](docs/COMMUNITY_BENCHMARKS.md#comparing-with-llamacpp)).

## How much RAM it keeps

Since 0.1.45-poly.7 the engine of the thirteen models keeps less of your RAM: it no longer holds the parts of the
model file that are already on the graphics card. On the laptop, Qwen3.6-35B-A3B after a 32,000-token prompt held
17.9 GiB with 0.1.45-poly.6 and holds 13.1 GiB now, and writes as fast. Under Linux it went from 17.8 to 12.6 GiB.

To hold less, add `"--ram-experts-mib", "6144"` to the `"args"` list of `strata-<model>.json`
([where](docs/INSTALL.md#where-things-are-stored)) and restart. The engine then holds that many MiB of RAM for the
experts that stay outside VRAM, while it answers too, and says so when it starts. The others stay in the system's
file cache, where your other programs can take that memory, and are read from it when a token needs them. Where the
experts fit in the RAM that is free it is off by default, because it writes slower, and we are still working on
that; where they do not, the engine turns it on by itself (below). On the laptop (RTX 4070 Laptop 8 GB, 32 GB of
RAM, Windows 11), Qwen3.6-35B-A3B UD-Q3_K_M:

| | default | `--ram-experts-mib 6144` |
| --- | ---: | ---: |
| RAM the engine holds after it loads | 11.7 GiB | **6.5 GiB** |
| at most while it writes: short chat / after a 32K prompt | 14.3 / 14.0 GiB | **6.7 / 7.3 GiB** |
| between requests | 12.8 GiB | **7.1 GiB** |
| Writes, tokens/s: the first short chat | 71.6 | 69.4 (3% slower) |
| after the 32K prompt | 65.4 | 63.7 (3% slower) |
| a short chat after that | 84.2 | 72.9 (13% slower) |
| a long answer (1,024 tokens of a short chat) | 90.6 | 78.3 (14% slower) |

Other programs held part of the card's VRAM in this run, so both columns write slower than with the card free. In
two other runs it wrote 6 to 8% slower after the 32K prompt and 8 to 10% slower in the chat after it. Under Linux
(an RTX PRO 6000 held to about 8 GB) it held 7.0 to 7.7 GiB instead of 12.3 to 14.3 and wrote 7 to 14% slower. With
the expert cache held still, as a release is checked, the answers are the same token for token.

When the experts outside VRAM do not fit in the RAM that is free, the engine now holds the ones used most in a
budget of that RAM less 4 GiB, and reads the others from the SSD when a token needs them; it says so when it starts.
Before, it mapped them all and the system read them back a page fault at a time. A container's memory limit counts
too. Measured under Linux on an RTX PRO 6000 held to about 8 GB, with the engine held to 12 GB and to 10 GB of RAM,
what a 16 GB PC has left for it, and the model file out of the file cache at each start: Qwen3.6-35B-A3B
UD-Q3_K_M, tokens/s in the first short chat, after a 24,000-token prompt and in a short chat after that.

| RAM for the engine | every expert mapped (before) | the budget it takes now |
| --- | ---: | ---: |
| 12 GB | 51 / 75-78 / 94-98 | **57-60 / 77-85 / 131-139** |
| 10 GB | 39 / 48 / 69 | **51 / 64 / 92** |

With no limit the same machine writes 63 / 169 / 203. It was not measured on Windows, where the same code runs.
`"--ram-experts-mib", "-1"` keeps every expert mapped as before.

The other way round, `"--lock-experts"` keeps the experts outside VRAM in RAM, so the system cannot give their
memory to other programs and have the engine read them from the SSD again later. On one NVIDIA card the engine
copies them into pinned memory at the load, which the system does not page out and `ulimit -l` does not limit, and
an expert that leaves VRAM later is copied in from the model file between requests. Where it makes no such copy (an
AMD card, a Mac, layers split over cards, too little free RAM) it locks the model file's pages instead; under Linux
that stops at `ulimit -l`, and the engine's start line gives the limit and how to raise it.

The engine also needs 350 to 400 MiB less VRAM for reading prompts: the expert cache takes it (Qwen3.6 keeps 2,712
experts in VRAM instead of 2,444), or `--vram-reserve-mib` leaves it to other programs.
How it works and how it was measured: [docs/BACKENDS.md](docs/BACKENDS.md#less-of-the-pcs-ram-and-vram).

## What you need

| Model | Graphics card | RAM (the installer's size) | Disk | System |
| --- | --- | --- | --- | --- |
| [Qwen3.8-Flash-Next](docs/models/qwen38.md) | NVIDIA RTX 20 to 50 series, or AMD RX 6800 / 6900, RX 7700 XT / 7800 XT / 7900 XT / XTX, RX 9060 XT / 9070 / 9070 XT, Radeon AI PRO R9700; 12 GB or more | 32 GB or more; 64 GB runs every size | about 80 GB | Windows, Linux |
| [GLM 5.3 Flash](docs/models/glm.md) | NVIDIA RTX 20 series or newer; only run on a 32 GB card | 96 GB beside a 32 GB card | 79 to 114 GB, on an SSD | Linux |
| [DeepSeek V4 Flash](docs/models/deepseek.md) | same as above | 72 GB beside a 32 GB card | 91 GB, on an SSD | Linux |
| [LFM2.5-24B-A2B](docs/models/lfm25.md) | NVIDIA RTX 20 series or newer, 8 GB or more | about 20 GB beside an 8 GB card | 11 to 25 GB | Windows, Linux, Mac (beta) |
| [Flint 40B-A2.5B](docs/models/flint.md) | same as above | about 22 GB beside an 8 GB card | 18 to 42 GB | Windows, Linux |
| [Qwen3.6-35B-A3B](docs/models/qwen36.md) | same as above; AMD as a beta | about 21 GB beside an 8 GB card | 13 to 23 GB | Windows, Linux, Mac (beta) |
| [Clover-35B-A3B-Preview](docs/models/clover.md) | NVIDIA RTX 20 series or newer, 8 GB or more | about 20 GB beside an 8 GB card | 13 to 21 GB | Windows, Linux |
| [Ornith 1.5](docs/models/ornith.md) | same as above; AMD as a beta | about 22 GB beside an 8 GB card | 14 to 24 GB | Windows, Linux |
| [Gemma 4 26B-A4B](docs/models/gemma4.md) | same as above; AMD as a beta | about 18 GB beside an 8 GB card | 11 to 17 GB | Windows, Linux |
| [Tiel Coder 35B-A3B](docs/models/tiel.md) | same as above; AMD as a beta | about 21 GB beside an 8 GB card | 14 to 23 GB | Windows, Linux |
| [MiniCPM-V 4.7 35B-A3B](docs/models/minicpmv.md) | same as above; AMD as a beta | about 25 GB beside an 8 GB card | 16 to 29 GB | Windows, Linux |
| [Kolibri 1](docs/models/kolibri.md) | NVIDIA RTX 20 series or newer, 8 GB or more | about 33 GB beside an 8 GB card | 29 to 48 GB | Windows, Linux |
| [Groot 19B-A3B 261023](docs/models/groot.md) (experimental) | shared CUDA/HIP backends; Q4_K_M text run on an 8 GB RTX 5060 (measured peak 7.05 GiB VRAM, 21.8 GiB RAM) | about 17 GB beside an 8 GB card for Q4_K_M, setup's fit estimate | 12.5 to 20.2 GB; images add 0.9 GB | Windows, Linux; images, Q6_K and Q8_0 untested |
| [Inkling 2 Mini](docs/models/inkling.md) (experimental) | NVIDIA, 8 GB or more; it has run on an RTX 5090 and on an RTX 4070 Laptop (8 GB, UD-Q3_K_XL) | about 21 GB beside an 8 GB card for UD-Q4_K_XL, setup's estimate | 9 to 18 GB | Windows, Linux, Mac (beta) |
| [Hy4-Flash](https://github.com/Niko1221/Strata/releases/tag/v0.2.18) (the Strata project's engine) | NVIDIA V100 to RTX 50, 8 GB or more; AMD RX 6000 to 9000 and Strix Halo; Intel Arc | 64 GB or more for SQ-64, 96 GB for SQ-96, 128 GB for SQ-128 | 130 to 181 GB, on an SSD | Windows, Linux, Mac |

RAM is the installer's estimate; a larger card needs less. The installer sets up everything else: on Windows it downloads the engine ready-made, and compiles it on your PC only on a processor without AVX2, for PolyStrata's own models on an AMD card, or when your copy's engine is newer than the latest release, offering to install the tools. Two or three cards can share a model ([multi-GPU](docs/MULTI_GPU.md)). Older graphics cards, Intel Arc, Apple Silicon Macs (Qwen3.8-Flash-Next, and Qwen3.6, LFM2.5 and Inkling 2 Mini as a beta), Strix Halo and older processors without AVX2: [the full list](docs/INSTALL.md#what-you-need). Each model's card, RAM and sizes in detail are on its page.

What Qwen3.6, Clover, Ornith 1.5, Gemma 4, Tiel Coder and MiniCPM-V each need and how it was checked: [docs/BACKENDS.md](docs/BACKENDS.md#install).

## Install

### Let your AI set it up

Do you use an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Paste this into it:

```text
Set up PolyStrata on this PC for me: https://github.com/simplexai-labs/PolyStrata - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk and picks the model that fits. Then it installs and starts it and tells
you how to connect your apps. AI tools can also install, start and stop PolyStrata through its
[MCP server](docs/MCP_SERVER.md).

### Or do it yourself

[Download PolyStrata](https://github.com/simplexai-labs/PolyStrata/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
**Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`** in the PolyStrata folder.

**Mac (Apple Silicon, M1 to M5, macOS 15 or newer):** install Apple's Command Line Tools once
(`xcode-select --install` in Terminal; Xcode itself is not needed). Then open Terminal in the PolyStrata folder and run
**`./setup.sh`**. It builds the Mac engine on your Mac (about 7 minutes), asks the same questions, downloads the model
and starts it. Next time, `./setup.sh` starts it right away. Which size fits your Mac and how fast it is:
[docs/MAC.md](docs/MAC.md). By default it sets up Qwen3.8-Flash-Next. `./setup.sh --family qwen36` or `--family lfm25`
sets up Qwen3.6 or LFM2.5 instead, on strata-poly's Metal engine, as a beta: on an M5 Ultra, Qwen3.6 UD-Q2_K_XL wrote
134 tokens/s in a short chat and read a 3,768-token prompt at 2,978 tokens/s. `--family inkling` sets up Inkling 2
Mini the same way: on the M5 Ultra, UD-Q3_K_XL wrote 74 tokens/s in a short chat, read a 32,787-token prompt at 2,440
tokens/s and wrote 61 tokens/s after it; it reads pictures on the CPU. Setup also offers GLM, DeepSeek, Ornith, Gemma
4, Tiel Coder and MiniCPM-V on a Mac, but they have not been run there with their real weights. Kolibri, Flint and
Clover need an NVIDIA or AMD card.

The installer finds your card and sets up the right engine for it. It asks you a few questions:

- which model and which size,
- how much context (how much text the model keeps in mind),
- whether it should read pictures (GLM 5.3 Flash, DeepSeek V4 Flash, LFM2.5, Flint, Clover and Kolibri 1 read none).

Press Enter each time for the recommended answer. Then it downloads the model (about 70 GB for Qwen3.8-Flash-Next,
79 to 114 GB for GLM 5.3 Flash, 91 GB for DeepSeek V4 Flash, 29 to 48 GB for Kolibri 1, 11 to 42 GB for the other eight) and starts it. If the download stops, run it again:
it continues where it left off. When the model is ready, its API is at `http://127.0.0.1:8080/v1`.
If you want an app to chat with it, we recommend [Linger](https://github.com/simplexai-labs/linger). Any app that
speaks the OpenAI or Anthropic API works too.

All fourteen are in the installer's list of models. `./setup.sh --family glm`, `deepseek`, `lfm25`, `flint`, `qwen36`, `clover`, `ornith`,
`gemma4`, `tiel`, `minicpmv`, `kolibri`, `groot` or `inkling` goes straight to one ([what it asks and writes](docs/GLM.md#install)).

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time).
> PolyStrata loads 35-55 GB into your RAM and locks part of it for the graphics card. This is normal. Wait, and don't
> close the window. The window shows what PolyStrata is doing.

**Next time**, run `START-HERE.bat` (or `./setup.sh`) again. It starts right away and downloads nothing twice. Close
its window to stop the model. `UPDATE.bat` (`./setup.sh --update`) updates PolyStrata without starting it.
Coming from 0.1.45-poly.10 or older: setup's files moved into `terminal/` ([what to change](docs/INSTALL.md#coming-from-0145-poly10-or-older)).
Updating, several cards, where the files go and every option: [docs/INSTALL.md](docs/INSTALL.md).

**Is it fast on your PC?** Run `tools/BENCH.bat` (`tools/bench.sh`): about 5-10 minutes, it measures chat, 4K/16K prompts and an agent-style session with your own settings and writes a report to share ([how](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

## Which model should I pick?

The same model comes in several sizes, compressed more or less: smaller sizes are faster and need less RAM, larger ones are closer to the original. The installer takes one for your RAM and card when you press Enter; for another size add `--model <size>` when you install.

| Model | The installer takes | With less RAM | By name |
| --- | --- | --- | --- |
| [Qwen3.8-Flash-Next](docs/models/qwen38.md) | 32 GB of RAM: Coder; 48 to 64 GB: IQ2_XS; 96 GB or more: IQ3_S |  | Q2_0, IQ3_XXS, Unsloth's 4- to 8-bit files |
| [GLM 5.3 Flash](docs/models/glm.md) | 3.0BIT | UD-IQ2_XXS, then REAP50-Q3_K_M |  |
| [DeepSeek V4 Flash](docs/models/deepseek.md) | UD-IQ2_XXS (the only size) |  |  |
| [LFM2.5-24B-A2B](docs/models/lfm25.md) | UD-Q4_K_XL | UD-Q3_K_XL | QAD-Q4_0, UD-Q6_K_XL, Q8_0 |
| [Flint 40B-A2.5B](docs/models/flint.md) | UD-Q4_K_XL from about 40 GB of RAM | UD-Q3_K_XL | UD-Q6_K_XL, Q8_0 |
| [Qwen3.6-35B-A3B](docs/models/qwen36.md) | UD-Q3_K_M | UD-Q2_K_XL | UD-Q4_K_M |
| [Clover-35B-A3B-Preview](docs/models/clover.md) | Q3_K_M | APEX-I-NANOPLUS | Q4_K_M |
| [Ornith 1.5](docs/models/ornith.md) | APEX-I-COMPACT | APEX-I-MINI | APEX-I-QUALITY |
| [Gemma 4 26B-A4B](docs/models/gemma4.md) | UD-Q3_K_M | UD-Q2_K_XL | UD-Q4_K_M, QAT-UD-Q4_K_XL |
| [Tiel Coder 35B-A3B](docs/models/tiel.md) | UD-Q3_K_XL | UD-IQ3_XXS | UD-Q4_K_XL |
| [MiniCPM-V 4.7 35B-A3B](docs/models/minicpmv.md) | Q4_K_M | mix_q3gud_im | Q6_K |
| [Kolibri 1](docs/models/kolibri.md) | Q4_K_M from about 48 GB of RAM | Q2_K | Q3_K_M |
| [Groot 19B-A3B 261023](docs/models/groot.md) | Q4_K_M | — | Q6_K, Q8_0 |
| [Inkling 2 Mini](docs/models/inkling.md) | UD-Q4_K_XL | UD-Q3_K_XL | UD-Q2_K_XL |
| [Hy4-Flash](https://github.com/Niko1221/Strata/releases/tag/v0.2.18) | SQ-64 from 64 GB of RAM, SQ-96 from 96 GB, SQ-128 from 128 GB |  |  |

What each size is, its download, the RAM it needs beside other cards and how close it comes to the largest: each model's page, linked from the table under "How fast is it?".

## Using it

All fourteen models are used the same way, through the same API.

<p align="center"><a href="https://github.com/simplexai-labs/linger"><img src="docs/media/linger-desktop.jpg" width="900" alt="Linger on a Mac: a chat room where agents report their work, beside the Chrome windows they work in"></a><br>
<sub><a href="https://github.com/simplexai-labs/linger">Linger</a>: several agents share one computer and the browser you are already signed in to. A picture from the Linger project.</sub></p>

- **As your personal agent:** we recommend [Linger](https://github.com/simplexai-labs/linger). Download it for Mac
  (Apple silicon), Windows or Linux from [its latest release](https://github.com/simplexai-labs/linger/releases/latest).
- **In a terminal:** `python polystrata.py --model <name>` in the PolyStrata folder chats with an installed model, and
  `--list-models` lists them ([more](docs/INSTALL.md#the-terminal)).
- **In the browser:** `http://127.0.0.1:8080` has a live **Monitor** of the model and your GPU/CPU/RAM,
  **Requests**, which opens the [API request monitor](docs/DETAILS.md#api-request-monitor) in a tab of its own once it is
  turned on, and **About** with the settings and addresses. With more than one model installed, the list at the top
  switches between them.
- **Your apps and coding agents:** add an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8080/v1`**. Any API key and any model name work.
  - Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, and `"anthropic_thinking": "on_request"` in `strata-<model>.json`
    ([why](docs/FAQ.md#can-i-use-it-with-claude-code-codex-or-other-apps))).
  - Codex CLI and other apps that use the OpenAI Responses API: `/v1/responses`
    ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Thinking:** choose **off, low, medium or high** in your app's "reasoning effort" setting. Off is the
  fastest. High is best for hard questions.
- **From your phone or another PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.

Where they differ:

| | Qwen3.8-Flash-Next | GLM 5.3 Flash | DeepSeek V4 Flash | LFM2.5-24B-A2B | Flint 40B-A2.5B | Kolibri 1 | Qwen3.6-35B-A3B, Clover-35B-A3B-Preview, Ornith 1.5, Gemma 4 26B-A4B, Tiel Coder 35B-A3B, MiniCPM-V 4.7 35B-A3B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Pictures** | Say yes to "Images?" in setup. Then attach pictures in your app. With less than 24 GB of graphics memory the picture encoder runs on the processor, on NVIDIA and AMD alike. | Not yet. | It reads text only. | It reads text only. | It reads text only. | It reads text only. | Say yes to "Images?" in setup, then as with Qwen3.8-Flash-Next. On a card with less than 16 GB the picture encoder runs on the processor, and a picture takes a few seconds. Clover reads text only. |
| **Several requests** | One at a time by default, the others wait. `"parallel": 2` answers several at once ([BATCHING.md](docs/BATCHING.md)); on a 12 GB card each answer gets slower. | One at a time, the others wait. | One at a time, the others wait. | One at a time, the others wait. | One at a time, the others wait. | One at a time, the others wait. | One at a time, the others wait. |
| **Several conversations** | Several are kept in RAM by default, room permitting, so follow-up messages and agents that take turns start in seconds ([details](docs/DETAILS.md#conversation-cache)). | The last one is kept. Setup asks whether to keep several in RAM, so that an agent and its subagents take turns without their prompts being read again ([how](docs/GLM.md#several-conversations-at-once)). | As GLM 5.3 Flash. | As GLM 5.3 Flash. | As GLM 5.3 Flash. | As GLM 5.3 Flash. | As GLM 5.3 Flash. |
| **Long prompts** | The first message of a chat is read in full, about 1 minute per 30,000 tokens. | The first message is read in full, about 40 seconds per 30,000 tokens. | The first message is read in full, about 29 seconds per 30,000 tokens. | The first message is read in full: 30,000 tokens take 1.3 to 1.5 seconds on the workstation and 8 to 10 seconds on the 8 GB laptop. | The first message is read in full: 30,000 tokens take about 2.3 seconds on the workstation and 15 to 22 seconds on the 8 GB laptop. | The first message is read in full: 30,000 tokens take about 6 seconds on the workstation with Q2_K and about 40 seconds on the 8 GB laptop. | The first message is read in full: 30,000 tokens take 4 to 5 seconds on the workstation and 14 to 27 seconds on the 8 GB laptop. |

More: [where things are stored](docs/INSTALL.md#where-things-are-stored), [the API](docs/DETAILS.md#using-it).

### Optional: more speed on one card when the whole model fits in RAM

Two opt-in settings (by RubenTorney, #1904 and #1905). They help when you run one graphics card and your RAM holds
every expert (PolyStrata's arena mode, for example Q2_0 or IQ3_XXS on a 64 GB PC). Open your config,
`strata-<model>.json` in the `terminal` folder of PolyStrata ([where](docs/INSTALL.md#where-things-are-stored)), and add:

- **Faster decode:** `"--adapt-async", "1"` to the `"args"` list. The expert cache then updates on a helper thread
  instead of pausing the answer. Measured on an RTX 5070 (Windows, Q2_0): +4-5% decode, +7-10% on code and long
  documents. Answers can differ slightly from the default (they stay the same from run to run).
- **Faster prompt reading on Windows:** `"STRATA_PIN_BY_NEED": "1"` in the `"env"` block. Windows can lock only about
  half the RAM for the graphics card; this locks the experts the card actually needs first. Measured on the RTX 5070:
  +4.4% prompt reading.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Restart PolyStrata (`START-HERE.bat` or `./setup.sh`) after the change. To undo, remove the two entries.

## Something went wrong?

- **My PC froze the first time PolyStrata started.** This is normal while it loads the model. Wait, and don't close the
  window. Still frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Run `START-HERE.bat` (or `./setup.sh`) again. It continues where
  it stopped.
- **It's very slow and the disk light keeps blinking, or it says "the engine stopped unexpectedly".** Your PC does
  not have enough free RAM. Close other programs (browsers use a lot), or pick a smaller size (Q2_0 or IQ2_XS).
- **It says port 8080 is already in use.** PolyStrata is already running. Look for its window.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Open an
[issue](https://github.com/simplexai-labs/PolyStrata/issues) and attach `strata-<model>.log` from the `terminal` folder. Found a
security problem? Report it privately: [SECURITY.md](SECURITY.md).

## How does it work?

Models like these usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
8-32 GB. PolyStrata makes the model fit by **sharing the work across your whole PC**. Think of a kitchen: the things
you use all the time stay on the counter, and the rest waits in the pantry.

Each model's diagram and how it works: [docs/HOW_IT_WORKS_MODELS.md](docs/HOW_IT_WORKS_MODELS.md).

## Built on PolyStrata

<p align="center"><a href="https://github.com/simplexai-labs/linger"><img src="https://raw.githubusercontent.com/simplexai-labs/linger/main/assets/linger-story.webp" width="820" alt="Linger in 28 seconds: a laptop opens, sets up a local model and connects apps. Mail, chats and meetings become shared memory. One request assembles an agent team; an agent uses the browser and waits for approval before sending an email. A timely reminder leads to a calendar booking, then the laptop closes."></a><br>
<sub>Linger in 28 seconds: it sets up a local model, connects your apps, turns mail and meetings into memory, and asks before it sends an email.</sub></p>

**[Linger](https://github.com/simplexai-labs/linger)** is an open-source personal agent that runs fully on your PC,
with its local models on PolyStrata. It finds what you need across your apps, remembers what matters and gets
things done while you are away. [Docs](https://docs.heylinger.com/) · [GitHub](https://github.com/simplexai-labs/linger)

Built something on PolyStrata? Open an [issue](https://github.com/simplexai-labs/PolyStrata/issues) and we will add it here.

## Credits and license

Qwen3.8-Flash-Next is by the [Qwen team](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). It was compressed by
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5) and Unsloth.
GLM 5.3 Flash runs from the files of
[neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF), Unsloth and patrickbdevaney.
Qwen3.6-35B-A3B is by the Qwen team, Gemma 4 by Google and DeepSeek V4 Flash by DeepSeek, all three in
[Unsloth](https://huggingface.co/unsloth)'s quantization. LFM2.5-24B-A2B is by
[Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), in Unsloth's quantization and in Liquid AI's own
QAD-Q4_0. Flint 40B-A2.5B is by [Recursive](https://huggingface.co/RecursiveSI/flint), in Unsloth's quantization.
Inkling 2 Mini is by [Thinking Machines](https://huggingface.co/thinkingmachines/Inkling-2-Mini), in Unsloth's
quantization. Groot 19B-A3B 261023 is by GrootTeam, in [bartowski](https://huggingface.co/bartowski/GrootTeam_Groot-19B-A3B-261023-GGUF)'s quantization.
Kolibri 1 is by [Aleph Alpha](https://huggingface.co/Aleph-Alpha/Kolibri-1), in
[Hob-forge](https://huggingface.co/Hob-forge/Kolibri-1-GGUF)'s GGUF conversion.
Clover-35B-A3B-Preview is by [DiffuSpace](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview), in
[maternion](https://huggingface.co/maternion/Clover-35B-A3B-Preview-GGUF)'s quantization and IsValorum's APEX.
Ornith 1.5 is by ornith-ai, in
[mudler](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF)'s APEX quantization. Tiel Coder 35B-A3B is Ornith 1.5 in
[peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP)'s quantization;
[u/Sensitive_Song4219](https://www.reddit.com/r/LocalLLM/comments/1x22inc/comment/pf3r111/) found that it runs on PolyStrata.
MiniCPM-V 4.7 35B-A3B is by [OpenBMB](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf), in their own
quantization; [iMilnb](https://github.com/iMilnb) brought it to PolyStrata.
PolyStrata is built on
[Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors, and uses parts of
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). PolyStrata is open source under the [MIT License](LICENSE). A few
parts and every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).
