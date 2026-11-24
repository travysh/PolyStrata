# PolyStrata on a Mac (Apple Silicon, Metal)

This page is for people with an Apple Silicon Mac (M1 or newer). PolyStrata runs there on its own Metal engine: the NVIDIA kernels
are replaced by Metal ones (`engine/metal/`), and the host code and the server are the PC's. Intel Macs cannot run it.

> **On this page:** [What you need](#what-you-need) · [Install](#install) · [Defaults](#defaults-and-options) ·
> [Gyro](#gyro-s-and-gyro-m-agentionais-trellis-experts) · [Measured](#measured-m5-max-128-gb) · [Limits](#limits-this-version) ·
> [Switches](#switches) · [How it works](#how-it-works) · [History](#history--older-measurements)

## Quick start

Install Apple's Command Line Tools once (`xcode-select --install`; Xcode itself is not needed), then run `./setup.sh` in the
PolyStrata folder. It sets things up the way `START-HERE.bat` does on Windows and builds the engine on your Mac (about 1 to
15 minutes); there is no ready-made Mac engine in this repository's releases. `./setup.sh --list-models` lists every model
and size.

Pictures are on by default ([Limits](#limits-this-version)). The NVIDIA, AMD and Intel engines are unchanged.

**Status:** new. It was measured on three Macs: an M5 Max with 128 GB, where the model stays in memory, and an M4 Mac mini with
32 GB and an M1 Max with 34 GB, where the experts stream from the SSD. Other chips are untested, and reports help
([COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md)).

## Model coverage

The Mac installer and `setup.sh` read every family and quantization from the shared `setup.py` catalog.
The added choices are experimental until tested with their actual weights. The existing Qwen defaults stay the same.

These families use native Metal `strata-poly`:

| `--family` | Model family |
|---|---|
| `glm` | GLM 5.3 Flash |
| `deepseek` | DeepSeek V4 Flash |
| `qwen36` | Qwen3.6-35B-A3B |
| `ornith` | Ornith 1.5 35B-A3B |
| `gemma4` | Gemma 4 26B-A4B |
| `tiel` | Tiel Coder 35B-A3B |
| `minicpmv` | MiniCPM-V 4.7 35B-A3B |
| `lfm25` | LFM2.5-24B-A2B |
| `inkling` | Inkling 2 Mini |

List their quantizations and download sizes, or set up an existing LFM file without downloading model assets:

```sh
./setup.sh --list-models
./setup.sh --family lfm25 --model QAD-Q4_0 --gguf-dir /path/to/ggufs \
  --no-download-models --yes --no-start
```

`--no-download-models` requires complete local model, draft and (when requested) projector files. Python packages
and build dependencies are separate. Files retain the shared installer's filenames and pinned checksums.
`--family` selects the smaller engine; a setup without it keeps the existing Qwen3.8 choice and defaults.
Within a smaller family, the default is its smallest download; `--model` keeps an explicit choice. Inkling 2 Mini
takes UD-Q3_K_XL from 32 GB of unified memory and UD-Q2_K_XL below that (UD-Q4_K_XL has not been measured on a Mac),
and reads pictures on the CPU: its picture tower has no Metal kernels.

This is **experimental**. LFM2.5 UD-Q3_K_XL has passed the real-weight smoke test below, and it and Qwen3.6
UD-Q2_K_XL have answered prompts up to 7,000 tokens on an M5 Ultra; the other families and quantizations have not
been tested with full weights on Metal. There is no broad model-quality evaluation.
The measured Qwen3.8 numbers elsewhere on this page do not describe these families. The default smaller-model
context is 8,192 tokens.

The engine uses the shared PolyStrata architecture code, expert cache, CPU experts, conversations, batching and
speculative decoding. It speaks the existing server protocol; OpenAI and Anthropic APIs and the web app remain
in `serve/server.py`. `--parallel`, `--conversation-cache-mib` and `--vram-reserve-mib` can be written by setup.
`--vision yes` builds the picture reader, PolyStrata's own vision tower with Metal kernels (no llama.cpp in it),
and uses the family's projector. It reads Qwen3.6's, Gemma 4's and MiniCPM-V 4.7's pictures on the GPU
([Local validation](#local-validation) has what was measured). GLM uses native KDA,
hyper-connections and sparse latent attention. DeepSeek uses native compression, indexer and attention kernels.

The native implementation is in `engine/metal/src/poly/`. It covers floating-point projections, quantized GGUF
projections, expert routing, normalization and activation, attention with FP16/Q8 caches, rotary positions,
convolution, and the recurrent delta network with speculative snapshots. The GGUF block decoders come from the
same pinned ggml checkout as the CPU expert code. Where MetalPerformancePrimitives' `matmul2d` compiles (macOS 26
and newer) and `--metal-optimize` is on (the default), a prompt batch runs on the matrix units: the quantized
projections decode their weights into threadgroup memory as fp16 and multiply there, a dense projection of 128
rows or more is first turned into fp16 once, and prompt attention is a flash-style kernel for heads of 64, 128 and
256 values. Otherwise the older kernels run. Decode attention splits a token's keys among up to 32 threadgroups
and merges their results.

On an M5 Ultra, against the v0.1.45-poly.10 engine loaded at the same time and sent the same text: thinking off,
greedy, answers of up to 200 tokens, 12 rounds taking turns with 3 seconds idle before each request; `--tail-skip 0
--max-context 8192`, the expert profile in each model's pack, `--threads 10 --metal-optimize --metal-memory balanced`,
the draft on. The two engines gave the same answer in every round. Each cell is the range and, in brackets, the median,
in tok/s: prompt read / decode.

| Model | Prompt tokens | This release | v0.1.45-poly.10 |
|---|---:|---|---|
| Qwen3.6 UD-Q2_K_XL | 31 | decode 123.9-138.2 (135.2) | decode 125.9-145.0 (137.1) |
| | 670 | 1,485-1,703 (1,523) / 118.5-129.9 (124.4) | 1,438-1,623 (1,482) / 110.0-129.8 (123.8) |
| | 1,797 | 2,586-2,721 (2,598) / 110.4-130.1 (121.1) | 2,521-2,703 (2,553) / 111.3-127.2 (119.3) |
| | 3,767 | 2,947-3,065 (3,004) / 95.6-101.9 (98.8) | 2,970-3,045 (2,993) / 95.3-99.8 (98.0) |
| | 7,216 | 2,744-2,772 (2,755) / 79.8-89.4 (81.5) | 2,710-2,758 (2,732) / 77.3-89.0 (81.9) |
| LFM2.5 UD-Q3_K_XL | 29 | decode 152.2-156.7 (154.9) | decode 152.9-156.8 (154.9) |
| | 643 | 1,844-2,076 (1,884) / 147.2-160.1 (149.7) | 1,810-2,114 (1,890) / 147.6-162.0 (149.8) |
| | 1,785 | 3,197-3,688 (3,458) / 144.0-157.3 (148.2) | 3,414-3,727 (3,458) / 143.7-158.4 (148.1) |
| | 3,679 | 4,192-4,489 (4,280) / 133.0-137.7 (135.2) | 4,122-4,350 (4,257) / 134.9-137.6 (135.5) |
| | 6,828 | 4,361-4,494 (4,422) / 120.9-126.3 (122.8) | 4,329-4,444 (4,417) / 121.2-126.4 (122.5) |

The two engines read and write at about the same speed on this Mac (their ranges overlap). Requests sent back to back, with no
idle time between them, read the prompt at one of two levels depending on what the GPU did just before: the 1,797-token
prompt jumped between about 2,900 and about 3,850 tok/s, and both engines landed on the slower level as often. The table
uses the reads after an idle pause for that reason. Other Apple GPUs have not been measured.

For these checks, add `-DSTRATA_METAL_POLY=ON` to the engine build below and run `ctest -L poly`. Tests include
CPU-reference kernel comparisons, small generated GGUF fixtures for all five architectures, graph replay,
CPU experts, batch-versus-single generation, and the native server protocol with conversation park/restore.
Synthetic weights check execution and state handling, not language-model quality. Metal fences order copies
and fills against compute, and macOS host-memory telemetry supplies the conversation cache's admission checks.

### Smaller-model Metal defaults

For `strata-poly` families, optimized expert dispatch and balanced memory management are enabled by default.
Compatible Qwen35MoE models automatically use Q8 KV caching. No optimization flags are needed:

```sh
./setup.sh --family tiel --model UD-Q4_K_XL --context 32768 \
  --yes --no-start
```

- Optimized dispatch (`--metal-optimize`, on by default) submits cached GPU experts while CPU experts run, and avoids empty
  expert matrix-multiplication threadgroups. It keeps the existing dot-product arithmetic.
  `--no-metal-optimize` disables it.
- `--metal-memory balanced` (the default) grows KV buffers in 4,096-token steps, with 1,024 tokens ahead,
  up to the selected context, and budgets expert copies against available unified memory.
  It keeps 4 GiB of host-memory headroom in addition to the GPU reserve. Expert-cache growth waits
  for two request-boundary observations with at least 512 MiB extra headroom after the
  256 MiB refill cushion. Warming prioritizes
  uncached experts and stops under pressure. Audited kernels declare the buffers each submission
  needs; unaudited paths retain conservative residency.
  `--metal-memory legacy` selects the original allocation policy.
- Automatic KV selection uses Q8 when the model is Qwen35MoE with compatible head geometry,
  otherwise FP16. `--poly-kv q8_0` explicitly selects Q8; `--poly-kv fp16` selects FP16.
  Direct engine launches use `--kv` for these overrides.
  Q8 is supported for the Qwen35MoE families `qwen36`, `ornith`, `tiel` and `minicpmv`.
  Other architectures reject Q8 rather than silently substituting a format.

Setup uses explicit command-line choices first, then saved explicit settings, then these defaults.
Saved opt-outs survive setup reruns. A normal start does not rewrite the config; omitted controls use the
new engine defaults. The original `strata` and separate `strata-q36` engines
reject these installer options. These options carry no throughput guarantee; compare timings on the same
Mac, model and prompt. Q8 changes KV precision and must be evaluated for the intended workload.
Cache resizing under memory pressure can still pause requests; these defaults do not resolve that delay.
Experimental kernel variants remain manual and existing hardware checks remain in force.

After reading the local GGUF headers, setup reports the initial and maximum KV allocation estimates for
Qwen35MoE, including an enabled draft block and the base sequence plus any parallel slots. It reports recurrent
sequence state separately when the metadata permits it. Unsupported or missing geometry is shown as unknown;
additional scratch, speculative snapshots and caches are also unknown. This is not a total RAM estimate:
weights, expert copies, other applications and macOS need space too. Under the legacy policy, the full
maximum-context KV cache is allocated at load on Metal even for a short prompt. The balanced policy keeps
that maximum as the limit while starting with fewer rows. KV growth needs temporary space for
both old and replacement buffers. If that cannot fit, the request reports an allocation error; it
does not silently shorten the selected context. Explicit `--kv-whole` still allocates the full KV
capacity. A frozen expert split is never resized to make room for KV growth.

### Additional Qwen and ready-pack choices

`qwen`, `swift`, `coder`, `unsloth` and `nvfp4` use the original `strata` Metal engine. This includes
Qwen IQ2_XS; Swift IQ2_XS/IQ3_XXS/IQ3_S; Coder IQ1_M; Unsloth UD-IQ4_XS/Q8_0; and both NVFP4 packs.
Coder's 256-expert routers and caches retain their own geometry. Swift is repacked losslessly so its dense
weights and experts are available together; its PLE table remains in the original shard.

NVFP4 reads the shared installer's pinned ready-pack manifests. The expert bytes and separate gate/up/down
scales stay intact, and a bounded SSD cache supplies native Metal software decoding. There is no full-model
conversion to another quantization. Its prompt path uses small native windows; throughput is unmeasured.
Generated ready packs verify bounded reads and scale preservation for 256/512 experts. NVFP4 expert projections
match ggml's decoder for 1, 4 and 8 tokens. Neither published NVFP4 pack has been run end to end on this Mac.

`qwen36-strata` selects the separate `strata-q36` Metal executable, including UD-Q4_K_XL, UD-Q6_K_XL and Q8_0.
It keeps the original driver's MTP, parallel sessions, hybrid CPU experts and server protocol. A build-time
adapter reads `engine/src/qwen35/q36_engine.cu`, emits Metal kernels and substitutes native dispatch calls. CUDA-only
fused optimizations use equivalent unfused operations. This path is distinct from `qwen36` on `strata-poly`.
Its generated model tests compare graph/plain/MTP generation, hybrid MTP, conversation replay and two parallel
MTP slots. Reference checks cover recurrent state updates and split-K attention.

Earlier Mac builds of `strata-q36` gave 0 for every routed expert, so only the shared expert reached the output. With
that fixed, the real UD-Q4_K_XL file writes the same tokens as `qwen36` on strata-poly (greedy, a 40-token prompt, 64
tokens). It is experimental on a Mac. The warm-up before the first answer took 0.5 seconds, and the first start of a
newly built engine takes about 20 seconds more while Metal compiles its shaders. For Qwen3.6 on a Mac the `qwen36`
family (strata-poly) is the one to install.

On the M1 Pro validation machine, GLM's generated in-file draft executes through the existing speculative
protocol. DeepSeek's generated prompt crosses its 128-token compression boundary and restores a parked
conversation. These checks establish execution and state handling, not output quality on the published models.
Run the full native regression with `ctest --test-dir engine/build-metal --output-on-failure`; individual groups use
`ctest -L poly`, `ctest -L q36`, and the `nvfp4_parity` / `ready_stream` tests.

### LFM2.5 real-weight smoke test

On 2026-11-02, an Apple M1 Pro with 32 GiB unified memory and macOS 27.0.1 ran
`LFM2.5-24B-A2B-UD-Q3_K_XL.gguf` (10,947,100,928 bytes) through native `strata-poly` and the existing Python
server. Setup downloaded the file and verified SHA-256
`d54834c785c7e265f1cb7375e47aa6f55a0b7cfab75816254c4d2e0c82fd5efc`.

```sh
./setup.sh --family lfm25 --model UD-Q3_K_XL --context 2048 --vision no --yes --no-start --no-browser
./run-lfm25-ud-q3_k_xl.sh
```

The engine loaded in 20.8 seconds, using eight CPU threads. All 2,432 routed experts fit in its 9.5 GiB expert
cache. Requests used temperature zero and disabled thinking (`reasoning_effort: "none"` for OpenAI requests,
`thinking: {"type": "disabled"}` for Anthropic requests).

After warmup, the prompt "Write a 300-word story about a robot learning to garden." generated the requested
limit of 256 tokens: 442.3 ms prompt processing, 6,821.8 ms generation, **37.5 tokens/s decode**, and 7.266 seconds
total HTTP time. These are one request's server-reported timings and client wall time, not a repeated benchmark.
The story stopped at the token limit. The first request after loading took 4.115 seconds for 11 output tokens.

OpenAI and Anthropic requests both answered `17 + 25` with `42`. A code request returned
`def square(x): return x * x`; streaming returned `hello` and its completion marker. A tool request returned
`get_weather` with `{"city":"Istanbul"}` as a structured tool call; no external weather tool was executed.
This checks basic generation and API integration, not general accuracy, long contexts, or image inference.

## What you need

- **An Apple Silicon Mac (M1 or newer).**
- **macOS 15 or newer.** The engine keeps its buffers resident through a residency set, which macOS 15 introduced.
- **For a source build only:** Command Line Tools (`xcode-select --install`).
- **Enough unified memory for the pack.** The GPU maps the model's main file and keeps it in memory while the model runs. Metal
  lets one program keep about 3/4 of the memory resident (107.5 GB of 128 GB on the M5 Max).

| Pack | The file the GPU maps | Needs about (with prompt buffers, KV, draft layer) | Recommended from |
| --- | --- | --- | --- |
| Q2_0 | 37.6 GB | 50 GB | 96 GB; streams below that |
| IQ3_XXS | 47.0 GB | 60 GB | 80 GB (where IQ3_S does not fit) |
| IQ3_S (recommended; the default from 96 GB) | 54.8 GB | 68 GB | 96 GB |
| UD-Q4_K_XL (Unsloth; higher quality) | 82.6 GB | 95 GB | 128 GB |
| UD-Q5_K_XL, UD-Q6_K_XL (Unsloth) | 103.9 / 114.8 GB | streams from the SSD on 128 GB | 128 GB, streamed |
| Gyro-S (agentionai, one 58.5 GB file) | 29.7 GB (the other 28.8 GB is the n-gram table) | 43 GB | 64 GB; streams below that |
| Gyro-M (agentionai, one 92.0 GB file) | 63.2 GB (+ the 28.8 GB table) | 76 GB | 128 GB; streams below that |

- **The n-gram table.** Each pack also has a 28.8 GB n-gram table (the model's second file). It is read from the SSD a few rows at
  a time and does not count against the memory above.
- **Disk.** The original Q2_0 and IQ3 packs download roughly 66 to 84 GB, plus preparation and draft data. Unsloth packs are much
  larger and Gyro uses different files. A streamed model can also need a separate experts file (34 to 40 GB for the examples
  below). Check the selected pack rather than treating these sizes as a universal limit.
- **Setup recommends and never forces.** It recommends the pack by memory. A bigger pack chosen with `--model` is kept, with a
  warning about what it risks.

### A Mac whose memory cannot hold the pack (32 to 64 GB)

The engine streams the experts from the SSD. It decides this itself: it streams when the model's weights plus 12 GiB for
everything beside them (prompt buffers, conversation memory, the draft layer, macOS) do not fit Metal's share. Setup uses the same
rule.

- It keeps the most-used experts in an arena and reads the rest as each token needs them, with read-ahead for the next layer's
  likely experts. The arena is Metal's share, or the memory macOS reports as available less what it keeps for macOS (7 GB, or 15%
  of the RAM when that is more), whichever is smaller. It counts as free only what macOS can give without swapping, fills the
  arena in 1.5 GB steps and stops early rather than push macOS below its reserve.
- On a Mac with 64 GB or more, the arena, the Metal buffers and the host memory together stay under 60% of the RAM, and the
  filled arena is wired (`mlock`) so macOS does not compress it while the GPU is idle. `STRATA_METAL_WIRE_ARENA=0` turns the
  wiring off. `STRATA_METAL_ARENA_GB` sets the arena yourself, with a warning when it is over the cap. The engine prints the
  planned footprint (`planned resident footprint ... cap ...`) and the footprint after the fill.
- Setup writes a second copy of the experts (`<model>.experts`, 40 GB for IQ3_XXS) in which each expert's three matrices sit
  together, so the SSD reads an expert with one request. On the 32 GB M4 that is about 17% faster decoding and 30% faster prompts
  with the same tokens. Setup writes it only when the disk has room for it plus 10 GB, otherwise it warns, and it does not write
  it on a Mac with 96 GB or more.
- Setup skips the MTP draft layer on such a Mac. DFlash2 is still available for compatible packs. If neither drafter is
  available, the model generates one token per step. A streaming Mac with 96 GB or more keeps the MTP layer and writes `--spec 3`.
- Prompts stream in chunks of 4096 on a streaming Mac with 96 GB or more, and 1024 on a smaller one. `STRATA_METAL_PROMPT_CHUNK=N`
  sets it.
- The model's file must be on the internal SSD. An external disk is untested and likely much slower.

## Install

Setup builds the Metal engine on your Mac:

```sh
git clone https://github.com/simplexai-labs/PolyStrata && cd PolyStrata
./setup.sh                 # asks the model and size; --yes takes the recommended answers
```

What the first run does (each step is skipped when it is already done):

1. **Python.** It needs Python 3.10 or newer, but macOS ships 3.9. If that is all your Mac has, `setup.sh` downloads a standalone
   Python 3.12 (25 MB, pinned by SHA-256) into `terminal/.python`. No sudo, no Homebrew.
2. **Checks your Mac:** the chip, the memory, the macOS version, the Command Line Tools and the free disk.
3. **Python packages** into `terminal/.venv`. CMake and Ninja come from pip too, installed just before the build.
4. **Builds the engine** (`engine/build-metal/strata`) with Apple's metal-cpp headers (pinned by SHA-256).
5. **Downloads the model.** The download resumes if interrupted. `--gguf-dir` uses files you already have.
6. **Prepares the model.** Setup makes the pack (seconds) and prepares the MTP draft layer where it fits (about 5 GB of the
   original checkpoint). A streaming Mac below 96 GB skips MTP and makes the experts' copy (about 3 minutes on the M4).
   Auto also fetches DFlash2 for compatible Qwen3.8-Flash-Next packs. Q2_0 and IQ3_XXS get the stock drafter (1.2 GB).
   UD-Q4_K_XL and UD-IQ4_XS-v31 prefer the tuned drafter (0.34 GB from the Strata project's signed update feed) and can use the stock drafter
   if the tuned one is unavailable. The tuned drafter uses `--spec 6` and `--draft-conf 0.7`. It is based on
   incoai/Qwen3.8-Flash-Next-DFlash2 and licensed under Apache-2.0. IQ3_S selects MTP.
   If DFlash2 is missing, the server uses MTP when it is available. `./setup.sh --setup` can fetch the drafter later.
   [DFlash2 license notice](../terminal/data/dflash2-tuned/NOTICE.md)
7. **Writes `strata-<model>.json` and `run-<model>.sh` in `terminal/`, then starts the server.**

After that, `./setup.sh` or the model's `terminal/run-<model>.sh` starts it directly. The server is the same as on a PC: the OpenAI-compatible
API on `http://127.0.0.1:8080/v1`, the Anthropic API on `/v1/messages`, and the Monitor page on `http://127.0.0.1:8080/`.
`python polystrata.py --list-models` in the PolyStrata folder lists the installed models, as on a PC.

| Option | Meaning |
| --- | --- |
| `--list-models` | Every family and size setup offers on a Mac, and exit. |
| `--family NAME --model QUANT` | The model and its size, as `--list-models` names them. For Qwen3.8-Flash-Next: `Q2_0`, `IQ3_XXS`, `IQ3_S`, Unsloth's `UD-Q4_K_XL` / `UD-Q5_K_XL` / `UD-Q6_K_XL` (`--family unsloth`) and `Gyro-S` / `Gyro-M` (`--family gyro`). |
| `--vision yes\|no\|gpu\|cpu` | Picture reader. |
| `--context N`, `--port`, `--host`, `--api-key` | Server settings. |
| `--draft-vocab`, `--no-browser` | Draft vocabulary; do not open the browser. |
| `--data-dir`, `--models-dir`, `--gguf-dir` | Where files go, or where yours are. |
| `--setup`, `--no-start`, `--check`, `--build`, `--update` | Choose another model or settings; install only; only check the Mac; build the engine again; update without starting. |

The model list and the recommendation judge a Mac by its unified memory with the rules of
`terminal/strata_control/mac_fit.py`. A pack that does not stay in memory is read from the SSD, and a Mac under 16 GB of
unified memory is told the model does not fit.

## Defaults and options

The config uses the Mac defaults: `--spec 2` with MTP (best on mixed text; an explicit `--spec N` in the config's `args` is kept),
or `--spec 3 --spec-min-p 0.5` for the IQ3_S, Q2_0 and Coder sets (M5 Max, 5 interleaved pairs, same tokens, over `--spec 2`: IQ3_S story -0.9%,
code +2.5%, JSON +16.7%, agent turn +4.9%; Q2_0 story +0.4%, code +11.5%, JSON +6.3%, agent turn +1.9%; Coder story -2.5%, code +7.1%,
JSON +9.3%, agent turn +12.5%; the other sets are not measured),
and the Q8_0 hyper-connection read. `STRATA_HC_Q8=0` in the config's `env` gives the exact BF16 read. The Mac engine is
deterministic: two runs give identical numbers.

### Longer context (`--rope-scaling`)

`--rope-scaling linear|yarn` with `--rope-scale`, `--rope-freq-scale` and the `--yarn-*` options changes the RoPE table of the
attention, as on a PC. Flash-Next is trained to 262,144 tokens: `--context 524288 --rope-scaling yarn --rope-scale 2` gives it
512K. The table keeps the model file's frequency base, and the engine logs the table it built (`rope scaling yarn in the window
table: ...`). Without `--rope-scaling` the table is the original one, bit for bit. The DFlash2 drafter keeps its own unscaled
table: it only proposes tokens, so the scaling changes how many it gets accepted, not the output.

### Gyro-S and Gyro-M (agentionai's trellis experts)

The Mac engine runs [Gyro](GYRO.md). The routed experts are trellis-coded blocks (ggml types TQ2_T, TQK6, TQK7) in a
Hadamard-rotated basis, read from the published file unchanged. The kernels are in `engine/metal/kernels/quant.metal`, `moe.metal` and
`moe_mm.metal`. Setup offers Gyro-S and Gyro-M next to the other sizes (`./setup.sh --family gyro --model Gyro-S`), never as the
default, and says what the choice risks on your Mac's memory. The 28.8 GB n-gram table is inside the file and is read from the SSD
like every other pack's, so Gyro-S needs 58.5 GB of disk and about 43 GB of memory to stay resident.

Gyro-M runs on the same kernels (checked tensor by tensor) but no Mac tested here could hold it. `engine/metal/tests/gyro_parity.sh`
checks the decoder against the format definition and the expert kernels against a double-precision reference.
`STRATA_METAL_GYRO_TILES=0` turns off the dense-weight tiles.

## Measured (M5 Max, 128 GB)

**Measured on:** Apple M5 Max, 128 GB, macOS 27.2, Qwen3.8-Flash-Next packs, greedy, 200 generated tokens, PolyStrata engine version
not recorded. Decode tok/s for a short question and a story ("prompt1 / story").

| | Q2_0 | IQ3_XXS | IQ3_S |
| --- | --- | --- | --- |
| Setup's defaults (`--spec 2` with MTP, suffix drafter on) | **119.5 / 108.2** | **108.9 / 97.0** | - |
| `--spec 2` with MTP, suffix drafter off | 114.0 / 107.9 | 103.5 / 96.9 | 98.2 / 94.1 |
| No MTP (one token per step) | 87.4 / 87.3 | 78.7 / 78.6 | 74.5 / 74.3 |

The output with MTP is the same as without it, token for token. MTP only proposes tokens, and the full model checks each one.

Prompt reading (time to first token), on the needle-test prompts:

| | 4K tokens | 32K tokens |
| --- | --- | --- |
| Q2_0 | about 1,430 tok/s | about 1,460 tok/s |
| IQ3_XXS | about 1,260 tok/s | about 1,240 tok/s |
| IQ3_S | about 1,320 tok/s | about 1,140-1,190 tok/s |

The first prompt after the Mac starts is slower while macOS reads the model file (IQ3_S: 413 tok/s on the first 4K prompt).
llama.cpp on the same Mac with Q2_0: 56.6 tok/s decode on the story, 887 / 809 tok/s on 4K / 32K prompts.

**Unsloth packs on the same Mac.** IQ3_S and UD-Q4_K_XL are the recommended sizes. Decode is tok/s over 200 tokens (greedy, MTP);
prompts are the warm rate from the second request on.

| | IQ3_S | UD-Q4_K_XL | UD-Q5_K_XL (streamed) | UD-Q6_K_XL (streamed) |
| --- | --- | --- | --- | --- |
| Quality against Q8_0 (top-1 / KL) | 87.9% / 0.142 | 92.8% / 0.055 | not measured | not measured |
| Decode, short question, no MTP / MTP | 75.4 / 101.9 | 60.7 / 94.0 | - | - |
| Decode, story, no MTP / MTP | 74.8 / 94.7 | 59.7 / 73.7 | 28 / 46-51 (`--spec 3`) | not measured / 47-51 (`--spec 3`) |
| Prompt 4K / 32K | about 1,180 / 1,290 | about 1,150 / 1,290 | about 1,070 / 1,110 | about 1,020 / 1,020 |
| llama.cpp Metal, decode (story / code) | 51.1 / 44.1 | 48.4 / 47.3 | 45.1 / 43.2 | does not run (Insufficient Memory) |

UD-Q5 and UD-Q6 cannot stay resident: their repacked file (103.9 and 114.8 GB) is bigger than the largest buffer Metal gives
(86.6 GB). The experts stream from the SSD into an arena of about 65 to 67 GB. The cost is a fixed hand-shake per layer, not the
SSD, so a longer window pays (Q5 story: 28 tok/s without MTP, 41.8 with `--spec 2`, 50.6 with `--spec 3`).

**Measured on:** Apple M4 Mac mini, 32 GB, experts streamed from the SSD (arena 11.5 to 14.7 GB), one token per step (no MTP),
medians of 5 interleaved runs, tok/s.

| | Q2_0 | IQ3_XXS | llama.cpp, CPU only (Q2_0 / IQ3_XXS) |
| --- | --- | --- | --- |
| Decode, short question (200 tokens) | **14.1** | **10.0** | 6.3 / 4.1 |
| Decode, story (200 tokens) | **13.5** | **9.2** | 6.3 / 4.1 |
| Decode after a 6K-token document | **13.1** | **7.7** | 5.1 / 2.8 |
| Prompt, 4K tokens | **93.6** | **65.8** | 21.2 / 8.7 |

The greedy answers equal the M5 Max's (model in memory) on 10 of 10 test prompts. MTP was slower here, because the draft layer's
1.2 GB do more as expert cache. The arena's size matters most: with 5.7 to 9.3 GB (a Mac with less memory free) the same M4 read
about 8.3 tok/s on a story. On an M1 Max with 34 GB (Q2_0, expert cache at 64 slots, kernel fix #2348) decode was 7.9 tok/s and a 9,940-token prompt read at 186 tok/s.

**Measured on:** Apple M4 Mac mini, 32 GB, Gyro-S, experts streamed (arena 11 to 12 GB of the 25.8 GB of experts), no experts' copy,
one token per step, no draft layer. Decode story, 120 tokens, 5 interleaved pairs, medians: **9.14 tok/s** (8.70 with the
dense-weight tiles off). Prompt of 4,065 tokens: **109 tok/s**. The engine held 18.6 GB of the 32 GB. Against the CUDA engine on
the same Gyro-S file (5 text segments, 4,995 positions, RTX 3060) the Mac is as far from either CUDA run as the two CUDA runs are
from each other: top-1 agreement 95.4% against 95.8%, mean KL 0.0207 against 0.0186.

## Limits (this version)

The first request after a model loads also waits once while the engine builds its prompt kernels. On one M2 Ultra (a user's report,
#5369) the first word came after about 20 seconds, 9.2 s of it for the kernels; later requests started in under a second.

- **Images (vision):** on by default. `./setup.sh` builds the picture reader, `strata-vision`, beside the engine and asks
  "Should the model also read images?" (default yes). The reader is PolyStrata's own vision tower (`terminal/tools/vision/own`)
  with Metal kernels for the Qwen models', Gemma 4's and MiniCPM-V 4.7's towers; no llama.cpp or ggml is built into it. With
  `--gpu` it reads on the Mac's GPU, without it on the CPU, from the same program. A Mac always gets the GPU reader, whatever its
  memory. `--vision cpu` is still allowed and prints a note (M5 Ultra, Qwen3.6's projector, a 256-token picture: 0.85 s on the
  CPU, 0.03 s on the GPU). If the GPU reader cannot start, the server prints one line and starts
  it on the CPU. The tower's measurements on an M5 Ultra are under [Local validation](#local-validation).

  **Measured on:** Mac mini M4, 32 GB, Qwen3.8-Flash-Next BF16 projector, with the earlier reader (llama.cpp's mtmd on ggml's
  Metal backend; it took about 1.18 GB on the GPU and 1.05 GB on the CPU).

  | Picture | GPU reader | CPU reader |
  | --- | --- | --- |
  | 260 tokens (640x400) | 0.45 s | 50.7 s |
  | 972 tokens (1600x1200) | 2.4 s | 194 s |

- **Context:** measured up to 32K (needles, KL). Setup recommends 32K below 96 GB and 64K from 96 GB. Up to the model's 262K can
  be set, but it is not measured.
- **Several requests at once.** `--parallel N` keeps N sequences side by side in the engine, each with its own running state and K/V (0.65 GB each at a 36,864-token context). Measured on an M5 Max with IQ3_S, the MTP draft layer and `--spec 2`, N requests of 150 tokens sent at the same time, tokens a second over all of them: 1 request 85, 2 requests 95 to 102, 3 requests 115, 4 requests 119 to 125. One request alone decodes at 96 to 100; with 2, 3 and 4 at once each decodes at 53 to 61, 41 to 48 and 32 to 40. The engine uses 8.3, 12.0, 14.0 and 16.0 GB for 1 to 4.
- **Not ported:** the control vector (`--control-vector-scaled`) and the PC tuning (`--calibrate`).
- **Streaming** is measured at 32 GB only.
- **Families:** every installer choice is offered ([model coverage](#model-coverage)). On the original engine the original
  Qwen3.8-Flash-Next, Unsloth's UD-Q4/Q5/Q6_K_XL and agentionai's Gyro-S are measured; setup repacks Unsloth's shards into one
  file for the GPU, and a Gyro file is used as it is. Qwen3.6, LFM2.5 and Inkling 2 Mini run on strata-poly's Metal engine as a beta (measured
  in [model coverage](#model-coverage), Inkling 2 Mini in [its page](models/inkling.md)); Qwen3.6 also runs on `strata-q36` (`qwen36-strata`), experimental
  ([Additional Qwen and ready-pack choices](#additional-qwen-and-ready-pack-choices)). The other newly ported paths (Swift, Coder, NVFP4, GLM, DeepSeek and the rest) have
  not run on a Mac with their real weights.

### Quality

- **The Q8_0 hyper-connection read** is the default. Teacher-forced over 28 prompts (code, reasoning, chat, six languages,
  documents), about 6,000 positions per set, BF16 reference: mean KL 0.0007 and top-1 agreement 98.9% on the model's own text,
  99.1% on another model's text, perplexity unchanged. For scale, IQ3_XXS against Q2_0 is a KL of 0.094.
- **The fast prompt path** against reading the prompt through decode windows: mean KL 0.0004, top-1 99.4 to 99.6%.
- **Needles:** a code word hidden in 8K, 16K and 32K tokens at 10/50/90% depth was found 9 of 9 with Q2_0 and with IQ3_XXS.
- **Same answers as on a PC:** against the CUDA engine on a Tesla P100 (IQ3_XXS and IQ3_S) the first 200 tokens of a reply on the
  short question are identical; against the HIP engine on a Radeon AI PRO R9700 (Q2_0), 36 of 39 first tokens are identical. The
  differences are near-ties between the two best tokens, from a different floating-point order.

## Switches

For testing. All are off unless set; the defaults are described above.

| Variable | Effect |
| --- | --- |
| `STRATA_HC_Q8=0` | The BF16 hyper-connection read (exact) instead of Q8_0 |
| `STRATA_METAL_PROMPT=window` | Read the prompt through decode windows (slow, exact fallback; about 190 tok/s) |
| `STRATA_METAL_PROMPT_CHUNK=N` | Prompt chunk size (default 2048 resident; see streaming above) |
| `STRATA_METAL_MTP_Q8=0` | The draft layer's matrices in BF16 instead of Q8_0 |
| `STRATA_METAL_MTP_CHAIN=0` | One GPU submission per draft step instead of one per round |
| `STRATA_METAL_FUSE=0`, `STRATA_METAL_LAZY_FOLD=0` | Turn off two decode fusions (same tokens, slower) |
| `STRATA_METAL_PROMPT_EXPERTS=v3` | The M5 layout for the prompt's expert GEMMs. Fails on an M1 Max, so it is off there (default v2) |
| `STRATA_METAL_STREAM=1` / `=0` | Stream the experts from the SSD even when the pack fits / never (the default decides by size) |
| `STRATA_METAL_CONTIG=0` | Streaming: read the model file even when the experts' copy is there |
| `STRATA_METAL_ARENA_GB=N`, `STRATA_METAL_KEEP_FREE_GB=N` | Streaming: the arena's size, or the memory left to macOS |
| `STRATA_METAL_LEAVE_SHARE=0.4` | Streaming, 64 GB or more: the share of the RAM left to macOS when the footprint cap is set |
| `STRATA_METAL_WIRE_ARENA=0` | Streaming, 64 GB or more: leave the filled arena pageable (default: wired) |
| `STRATA_METAL_IGNORE_REPACK=1` | Start although a model repack or setup's preparation is running (the engine refuses by default) |
| `STRATA_METAL_KEEP_HOST_WEIGHTS=1` | Streaming: keep the host code's own copies of the weights (2 to 3 GB) |
| `STRATA_METAL_F16=1` / `=0` | Decode windows of a resident model: the GEMVs of Q2_0, Q3_K-Q6_K and IQ4_XS and the Q2_0 routed experts in fp16. On in the config setup writes for a resident IQ3_S or Q2_0 pack (M5 Max: decode +9 to +15%, top-1 96-97% and mean KL 0.009-0.016 against fp32; the 61-task quality suite passes the same tasks as with it off); `=0` in the config's `env` turns it off. Other sets: off unless set. M1 Max: a 2-token window's GPU time 56-60 -> 40-41 ms, mean KL 0.009 against fp32 |
| `STRATA_METAL_TRACE=1`, `STRATA_METAL_PROFILE=1`, `STRATA_METAL_KPROF=T` | Per-window traces, a time breakdown, per-kernel GPU times |
| `STRATA_MACOS_SENSORS=1` | The Monitor's GPU temperature and power without root (see below) |

### Temperature and power in the Monitor

Without a switch the Monitor shows no GPU temperature or power on a Mac: reading them takes root (`powermetrics`).
`STRATA_MACOS_SENSORS=1` reads them without root, the way the Stats app does: the temperature from the SMC (or the HID
sensors), the power from IOReport's energy counter. These are macOS's private calls, which Apple does not document and
can change in any release. A call that goes wrong inside them can stop the server, which is why this is off by default.
The SMC's temperature keys differ between chips, so the list of keys is a table by chip (M1 to M6 and the A18 Pro,
with the M4's base chip and its Pro and Max apart). It is copied from the Stats app's own table (commit `032f4a5e4`);
only the M5 has been run, on an M5 Pro, and its idle readings were also checked on an M5 Max. On another chip a key that is missing is skipped, and a chip with no keys, or none
that answer, shows the GPU temperature from the HID sensors where it has them, and none where it does not.

## Build and regression checks

The standalone checks generate their own inputs and do not need model files or metal-cpp:

```sh
cmake -S engine/metal -B engine/build-metal-check -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build-metal-check -j 4
ctest --test-dir engine/build-metal-check --output-on-failure
```

CTest registers the memory-budget check and kernel parity checks, and serializes GPU tests even with `ctest -j`.
The benchmarks and model-dependent chain programs are separate. To include the engine and runtime checks, configure
with `-DMETAL_CPP_DIR=/path/to/metal-cpp -DSTRATA_GGML_DIR=/path/to/llama.cpp`, using setup's pinned dependencies.
`-DBUILD_TESTING=OFF` disables CTest registration. A passing kernel suite does not establish model generation,
server behavior, or performance on other Apple chips.

Pipeline handles share ownership across copies and release the native Metal pipeline when the last copy goes
away. This avoids retaining every compiled pipeline after a model context is destroyed. `pipeline_lifetime`
checks dispatch through a surviving copy and release of the last owner; it is not a throughput benchmark.

### Local validation

On an M5 Ultra. A setup from scratch with the model files already on the disk (nothing downloaded) and pictures on took
43 to 51 seconds, and `python polystrata.py --list-models` then listed the model:

```sh
./setup.sh --family qwen36 --model UD-Q2_K_XL --gguf-dir <folder with the files> --vision yes --yes --no-start
```

`python polystrata.py benchmark --model <name>` on that config (setup's defaults for Qwen3.6 UD-Q2_K_XL, which include
`--tail-skip 6`): a 21-token prompt, first token after 0.23 seconds and the answer at 148 tok/s; a 2,981-token prompt,
read at 3,890 tok/s and the answer at 117 tok/s. The engine speeds against v0.1.45-poly.10 are in
[model coverage](#model-coverage).

Through the server with Qwen3.6 UD-Q2_K_XL, the answer about a screenshot of code gave the numbers in it, and a photo was
described correctly.

Pictures on the Metal tower against the Metal reader of v0.1.45-poly.10 (llama.cpp's mtmd on ggml's Metal backend), on
the GPU: a picture of 992 tokens, a 1600x1200 photo and a 1024x1024 picture, the time of each picture's second read in
milliseconds, the peak memory from `/usr/bin/time -l`, 3 rounds taking turns.

| Model | This reader | v0.1.45-poly.10 |
| --- | --- | --- |
| Gemma 4 | 922 / 675-677 / 238-239 ms, peak 993 MiB | 963-965 / 721-722 / 271 ms, 1,806 MiB |
| MiniCPM-V 4.7 | 174-175 / 174-175 / 134 ms, 1,596 MiB | 276-277 / 276 / 198 ms, 1,680 MiB |
| Qwen3.6 | 816-819 / 591-592 / 141 ms, 758-793 MiB | 898 / 672 / 234 ms, 1,567-1,584 MiB |

The reader now reads the weights in place from the mapped mmproj file. macOS can drop those pages and read them again,
so they do not count in the process's memory.

`tools/packaging/mac/build_engine.sh --version 0.1.63` made a Mac engine package of 6.8 MiB in 41 files that needs
macOS 15.0 or newer. These are checks on one Mac, not a quality evaluation.

## How it works

The code is in `engine/metal/` (the engine's side in `engine/metal/src/engine`, the kernels in `engine/metal/kernels`).

- **Host code:** the PC host code compiles unchanged for macOS. `engine/metal/include/metal_compat` maps the CUDA runtime calls it makes
  onto a small Metal runtime (`engine/metal/src/gpu`).
- **Decode and prompt work** run in Metal kernels: one command buffer per verify window, experts read in place from the mapped
  model file, hyper-connections, GDN state and QSA attention on the GPU, and a whole MTP draft round in one command buffer.
- **PC paths stay separate:** every shared file's change is behind `STRATA_GPU_METAL` or `__APPLE__`, and the Metal build has
  its own CMake project. Whether PolyStrata's PC builds come out byte-identical with this code present was not measured.

## History / older measurements

**Why `--spec 2`.** Measured on: Apple M5 Max, 128 GB, decode tok/s with MTP on the two prompts (prompt1 / story).

| Window | Q2_0 | IQ3_XXS |
| --- | --- | --- |
| `--spec 2` | 111 / 106 | 102 / 95 |
| `--spec 3` | 116 / 92 | 108 / 95 |
| `--spec 4` | 111 / 81 | 108 / 83 |
| `--spec 5-6` | slower | slower |

**The experts' copy.** Measured on: Apple M4 Mac mini, 32 GB, IQ3_XXS after a 6K-token document, arena held at 12 GB, medians of 5
interleaved runs, identical tokens in all 10 runs.

| | Decode | The 6K prompt | SSD wait per token |
| --- | --- | --- | --- |
| The model file (three reads per expert) | 7.37 tok/s | 64.7 tok/s | 46.6 ms |
| The experts' copy (one read per expert) | **8.62 tok/s** | **84.1 tok/s** | 31.5 ms |

The 4K prompt read 72.9 tok/s with the copy against 62.5 without. A 34-minute server run with the copy (47 requests, context
34,816) got every answer right and did not slow down. With the model file, an arena of 7.5 GB gave 5.5 tok/s, 10.5 GB gave 6.7 and
12 GB gave 7.5. On the 128 GB M5 Max the copy for Q5 did not pay (decode 50.3 against 50.2 tok/s). The SSD alone measures 2.97 GB/s
for one 1.75 MB read per expert against 2.30 GB/s for three 0.58 MB reads. The Q2_0 copy was not measured.

**Prompt chunk on a streaming Mac.** Measured on: M5 Max, UD-Q5, warm, 2 interleaved rounds. Chunks of 1024, 2048 and 4096 gave 700,
920 and 1,100 tok/s on the 4K prompt (32K: 710, 970, 1,100). Decode did not change. On a resident model 4096 instead of 2048 gained
only 2 to 4% (Q2_0), so that default stays.

**Memory footprint cap.** Measured on: M5 Max, 128 GB. Before the cap, a streaming arena of 100 GB compressed 19 GB of itself in 7
seconds, and the kernel watchdog panicked three times. With 40% of the RAM left to macOS and the arena wired, a 40 GB file write
beside a 16 GB program caused no swap-out. A resident pack (UD-Q4_K_XL, wired 91.7 GB) took the same load with nothing compressed,
so resident packs are limited to 70% of the RAM.

**Gyro-S kernel profile.** Measured on: M4 Mac mini, 32 GB. A decode window of two tokens takes 111 ms: the routed experts about
40 ms, the dense weights (Q5_K and Q6_K, 3.6 GB read a token) about 66 ms before the tile change. The kernel for one expert layer
(ten experts, one token) takes 278 us at 37.5 GB/s; IQ3_XXS takes 333 us and Q2_0 166 us. The trellis decode is limited by latency
and ALU, not by memory. The prompt kernels run 68 ms a layer against 62 ms for IQ3_XXS at 1,024 tokens.

**Quality against the CUDA engine, Gyro-S file.** `terminal/tools/quant_quality.py`, teacher-forced over 4,995 positions, CUDA = RTX 3060.

| Reference -> run | top-1 agreement | mean KL | KL p99 | perplexity |
| --- | --- | --- | --- | --- |
| CUDA GPU experts -> CUDA CPU experts (noise floor) | 95.80% | 0.0186 | 0.195 | 4.462 / 4.477 |
| CUDA GPU experts -> Mac | 95.46% | 0.0207 | 0.241 | 4.486 / 4.477 |
| CUDA CPU experts -> Mac | 95.38% | 0.0187 | 0.272 | 4.486 / 4.462 |
