# Inkling 2 Mini

Back to [README.md](../../README.md)

**Inkling 2 Mini** by Thinking Machines: a file of 9 to 18 GB. It is experimental here. It has run on Linux with an
RTX 5090, on Windows with an 8 GB laptop card and on a Mac with an M5 Ultra (a beta there). It
reads pictures too. Its 24 layers of weights run as 48: the twelve in the middle run three
times for every token. Forty of the 48 look back 1,024 tokens only, so a long conversation takes little VRAM:
32,768 bytes a token.

## How fast is it?

Measured on an RTX 5090 (32 GB) with a Xeon Platinum 8470Q (25 vCPU) and 90 GB of RAM, on Linux. The card holds
every expert of all three sizes. Each run writes 256 tokens, the most likely one each time, at a 32K context, and each
number is the range of three runs.

| Size | Writes after a 19-token prompt | Writes after a 10,081-token prompt | Reads that prompt |
| --- | ---: | ---: | ---: |
| **UD-Q4_K_XL** | 111-112 tokens/s | 109 tokens/s | 3,290-3,300 tokens/s |
| **UD-Q3_K_XL** | 114 tokens/s | 110-112 tokens/s | 3,360-3,450 tokens/s |

llama.cpp on the same card, run in turns with the above on the same prompts (commit 29b52d3 with the fix of its
pull request 32704, `llama-server -ngl 99`):

| Size | Writes after a 19-token prompt | Writes after a 10,081-token prompt | Reads that prompt |
| --- | ---: | ---: | ---: |
| **UD-Q4_K_XL** | 82-83 tokens/s | 76-77 tokens/s | 1,540-1,550 tokens/s |
| **UD-Q3_K_XL** | 84-86 tokens/s | 77-78 tokens/s | 1,510-1,530 tokens/s |

Both runs print the three likeliest tokens of every step, which the comparison below reads; that costs a little.
Without it, UD-Q4_K_XL writes 127-129 tokens/s after the short prompt and 124 after the long one, and reads the long
one at 3,280-3,320 tokens/s (six runs each); the README's benchmark measures 128.

The two start with the same tokens. For UD-Q4_K_XL the first 13 after the short prompt and the first 14 after the
long one are the same, and they part at a step whose two leading candidates are nearly as likely (0.35 against 0.34,
and 0.028 against 0.028). For UD-Q3_K_XL the first 13 and the first 40 are the same.

On an RTX 30 card or newer, UD-Q2_K_XL's MXFP4 experts (the down of one layer) are multiplied on the int8 tensor
cores while it reads a prompt, with the prompt's numbers rounded to 8 bits, as the file's other experts already are.
On an RTX 3090 that reads a 2,048-token prompt 3.1% faster (2,068 against 2,005 tokens/s) and an 8,192-token one 1.0%
faster. Against a run with every product in halves, it moved the answers no more than the other int8 products do on
a 302- and an 8,192-token prompt (one prompt each). `STRATA_PREFILL_MXFP4=fp16` keeps them on halves, as before.

On an RTX 50 card the engine multiplies the experts kept in NVFP4 (most of them, in both sizes) on the FP4 tensor
cores while it reads a prompt, with the prompt's numbers rounded to about 8 bits instead of 11. That reads a prompt
1.5 to 1.9 times as fast as multiplying them as halves (`STRATA_PREFILL_NVFP4=fp16`). It moves the answers a little:
written after the halves' greedy answers to eight prompts of 19 to 10,124 tokens, 256 tokens each, it gives the same
most likely token at 98.5% of the steps (95.3% to 100% a prompt); the halves run again give 100%. A card that holds
only part of the experts was not measured. Three runs each, 32 tokens written after the prompt:

| Size | Prompt | Halves (`fp16`) | FP4 (the default on RTX 50) |
| --- | ---: | ---: | ---: |
| **UD-Q4_K_XL** | 2,664 tokens | 2,450-2,490 tokens/s | 3,800-3,840 tokens/s |
| | 10,081 tokens | 3,200-3,220 tokens/s | 5,390-5,410 tokens/s |
| | 20,162 tokens | 3,250-3,260 tokens/s | 5,500-5,510 tokens/s |
| **UD-Q3_K_XL** | 2,664 tokens | 2,570-2,600 tokens/s | 3,750-3,880 tokens/s |
| | 10,081 tokens | 3,340-3,360 tokens/s | 5,360-5,400 tokens/s |
| | 20,162 tokens | 3,390-3,400 tokens/s | 5,480-5,510 tokens/s |

On an RTX 20 card (Turing) the engine multiplies the experts as halves while it reads a prompt. With `--prompt-int8`
it multiplies them on the int8 tensor cores, as it does on RTX 30 and newer, with the prompt's numbers rounded to 8
bits with a scale for every 32 (`STRATA_PF_I8MMA=0` turns the int8 products off on every card). It is not the
default because it moves the answers more: measured on a Quadro RTX 4000 (8 GB),
which holds a third to two thirds of the experts, written after the halves' greedy answers to three prompts of 302
to 8,192 tokens, 64 tokens each, with three files (the two below and an IQ3_S), it gives the same most likely token
at 560 of the 576 steps (97.2%). Prompt speed on that card, three runs each:

| Size | Prompt | Halves (the default on RTX 20) | int8 (`--prompt-int8`) |
| --- | ---: | ---: | ---: |
| **IQ4_NL** | 2,048 tokens | 329-338 tokens/s | 419-447 tokens/s |
| | 8,192 tokens | 355-361 tokens/s | 456-461 tokens/s |
| **UD-Q2_K_XL** | 2,048 tokens | 361-363 tokens/s | 434-439 tokens/s |
| | 8,192 tokens | 347-348 tokens/s | 411-412 tokens/s |

With `--prompt-int8`, an RTX 20 card also multiplies a prompt batch's experts in VRAM in one launch for each
class of rows (up to 32, up to 128, more) over all of that class's experts, wherever their slots are, in place of a
launch for each run of slots that lie one behind the other. The int8 kernel sums each output in the same order
whatever its launch, so the answers are the same to the bit. It was measured +1.8% to +2.5% on prompts on an RTX 2080
Ti (22 GB, every expert in VRAM) and no change on an RTX 2070 (8 GB). `STRATA_POLY_EXPERT_TILES=0` goes back to a
launch for each run.

On an RTX 20 card a prompt's attention runs in a tensor-core kernel shaped for Turing: one layer's call of 1,024 rows
takes 1.0 ms in a sliding-window layer and 18 ms in a full layer at 32K keys (RTX 2070 Super). It rounds the
attention's sums in another order than the kernel before it, which `STRATA_INK_MMA75=0` goes back to (2.8 ms and
39 ms; a 32K prompt reads about 9% slower). Against that kernel's greedy answers to the three prompts above (64 tokens
each, the three files) it gives the same most likely token at 566 of the 576 steps (98.3%).

On an RTX 30 card or newer, most of a prompt's products run on the int8 tensor cores, with the prompt's numbers
rounded to 8 bits. UD-Q2_K_XL's IQ1_S experts (the gate and up of 6 layers) are the exception: they stay on halves
unless `STRATA_PREFILL_IQ1S=int8` is set. On an RTX 3090 that setting reads a 2,048-token prompt 7.7% faster (2,159
against 2,005 tokens/s) and an 8,192-token one 6.7% faster, but on a 302-token prompt it moved the answers more for
each matrix than the other experts' int8 products do, so it is opt-in until more prompts are measured.

On a card that does not hold every expert, a prompt of more than one batch hands each call's row tables to the card
by a kernel, so they no longer wait behind the experts copied over the bus. The answers are the same and no VRAM is
added. On an RTX 2070 Super 8 GB a 2,048-token prompt reads 6.5% faster with IQ4_NL (477 against 448 tokens/s) and
2.7% faster with UD-Q2_K_XL; 32,768 tokens read as fast as before. `STRATA_INK_KERNEL_TABLES=0` turns it off.

On a laptop with an RTX 4070 Laptop (8 GB), a Core i7-14700HX and 32 GB of RAM, on Windows, UD-Q3_K_XL has 1,216 of
its 3,072 experts on the card and the rest are computed on the CPU. On a card that does not hold every expert, setup
gives Inkling 2 Mini three settings that read fewer bytes for each token: `--attn-requant q5_0` (the attention's
matrices in 5 bits instead of 8), `--loop-bf16` (the loop controllers in BF16 instead of F32) and `--tail-skip 4`.
A card that holds the whole model gets none of them. Expert swaps stay on: with `--expert-swaps 0` the experts the
card holds stay the ones it loaded, fewer of a token's experts are found there, and the CPU computes more of them
(16-23% slower writing after a 302-token prompt on an RTX 4060 Ti 16 GB, a Quadro RTX 4000 8 GB and an RTX 2070
Super 8 GB, about the same after 8,192). On the laptop, the fastest of three runs, 256 tokens written each; the last
column is how often the most likely token is the same as without the settings, written after their greedy answer to
the 4,526-token prompt:

| UD-Q3_K_XL | Writes after a 25-token prompt | Writes after a 4,526-token prompt | Same most likely token |
| --- | ---: | ---: | ---: |
| None of the settings | 37.8 tokens/s | 41.8 tokens/s | 99.6% (run again) |
| Setup's three settings | 47.7 tokens/s | 50.0 tokens/s | 98.0% |
| The three and `--expert-swaps 0` | 53.6 tokens/s | 52.7 tokens/s | 95.7% |

On the laptop `--expert-swaps 0` writes faster, but it moves the answers more, so setup leaves it out.

For answers closer to the model without them (a high-accuracy mode), `./setup.sh --family inkling
--inkling-small-card off` leaves the three settings out; a later setup run keeps that (`"small_card": "off"` in the
run config, also on a card that holds every expert) until `--inkling-small-card on`. With two cards it applies to the
one card Inkling 2 runs on. An existing run config can also be overridden by adding
`--attn-requant off --no-loop-bf16 --tail-skip 0` at the end of its arguments (the last of each flag counts;
`STRATA_ROUTE_TAIL_SKIP`, the app's "Skip rarely used parts of the model" switch, overrides `--tail-skip`). Writing
is slower on an 8 GB card this way (see the table above).

### Setup's four settings, one at a time

These settings make the answers worse to make them faster. Setup adds three of them by itself, without asking,
whenever the card cannot hold every expert of the size: `--attn-requant q5_0`, `--loop-bf16` and `--tail-skip 4`.
It no longer adds `--expert-swaps 0` (measured below because setup used to); `--inkling-small-card off` (above) leaves the
three out. `--attn-requant q5_0` stores the attention's matrices in fewer bits and `--loop-bf16` the loop
controllers: lossy rounding. `--tail-skip 4` does not compute some of a token's experts at all: on the 8 GB cards
below, with the other three settings, 9% to 27% of a token's expert picks. The answers differ measurably from the model's without them
(the same most likely token at 94.3-95.0% of the steps with all four, against 98.2-99.0% for a second run without
them), and with `--tail-skip` the engine's "experts from VRAM" figure counts the skipped experts as found in VRAM, so
it reads high.

To turn them off, delete `"--attn-requant", "q5_0", "--loop-bf16", "--tail-skip", "4"` from
the `"args"` list of the model's run config, `strata-inkling-<size>.json` in the install's `terminal` folder (for
example `strata-inkling-ud-q3_k_xl.json`), and restart the server. To keep the rounding but compute every expert,
change the `4` after `--tail-skip` to `0`, or add `"env": {"STRATA_ROUTE_TAIL_SKIP": "0"}` to the config. Running
setup again for the size writes the three back into `"args"`; an `"env"` entry you added stays.

Three of them round the numbers more coarsely (lossy). `--tail-skip 4` is different: it leaves experts out. For each
token, an expert that is not on the card and that the router ranks fifth or lower of the token's eight is not
computed at all. While a prompt is read, a group of tokens at a time, an expert is left out only if every token of
the group ranks it fifth or lower.

| Setting | What it does | Kind |
| --- | --- | --- |
| `--attn-requant q5_0` | the attention's matrices in 5 bits instead of 8: fewer bytes a token, and more VRAM for experts | rounding |
| `--loop-bf16` | the loop controllers in BF16 instead of F32 | rounding |
| `--expert-swaps 0` | no expert moves into VRAM while an answer is written (default 64 on their way at a time) | rounding: the same experts, but more of them on the CPU, which rounds a little differently |
| `--tail-skip 4` | a token's experts ranked 5th to 8th that are not on the card are left out | **skips computation** |

Measured on main d1f711e11, one setting at a time and all four, with ggml-org's IQ4_NL file and with UD-Q2_K_XL
(Unsloth's tensors byte for byte, in the `inkling2_loop` layout). 256 tokens written greedily at a 32K context with
an FP16 KV cache; each number is the median of two runs, in tokens/s:

| Card | Settings | Experts on the card | Writes after 302 tokens | Writes after 8,192 | Reads 2,048 |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 4060 Ti 16 GB, Threadripper 2950X | none | 2,619 of 3,072 | 49.0 | 44.6 | 997 |
| | `--expert-swaps 0` | 2,619 | 41.3 | 42.8 | 965 |
| | `--loop-bf16` | 2,632 | 49.9 | 45.5 | 1,040 |
| | `--attn-requant q5_0` | 2,666 | **54.8** | 49.4 | 1,040 |
| | `--tail-skip 4` | 2,619 | 50.9 | 46.2 | 1,011 |
| | all four | 2,678 | 53.9 | **52.5** | 1,019 |
| Quadro RTX 4000 8 GB, Ryzen 5 5600X | none | 998 of 3,072 | 30.9 | 21.7 | 341 |
| | `--expert-swaps 0` | 998 | 23.8 | 22.8 | 338 |
| | `--loop-bf16` | 1,008 | 29.6 | 20.2 | 346 |
| | `--attn-requant q5_0` | 1,045 | 30.9 | 19.6 | 355 |
| | `--tail-skip 4` | 998 | 31.0 | 19.9 | 352 |
| | all four | 1,055 | 30.9 | 18.3 | 358 |
| RTX 2070 Super 8 GB, Ryzen 5 3600XT, UD-Q2_K_XL | none | 1,986 of 3,072 | 39.3 | 32.7 | 515 |
| | `--expert-swaps 0` | 1,986 | 32.3 | 32.9 | 512 |
| | `--loop-bf16` | 2,008 | 40.0 | 33.4 | 515 |
| | `--attn-requant q5_0` | 2,039 | 43.6 | 34.8 | 517 |
| | `--tail-skip 4` | 1,986 | **43.7** | 37.4 | 516 |
| | all four | 2,061 | 43.0 | **40.5** | 515 |

On the 16 GB card `--attn-requant q5_0` brings most of the gain. With UD-Q2_K_XL on the RTX 2070 Super,
`--tail-skip 4` and `--attn-requant q5_0` bring about the same after 302 tokens (+4.4 and +4.3 tokens/s); after
8,192, `--tail-skip 4` brings +4.7 and `--attn-requant q5_0` +2.1. `--expert-swaps 0` costs speed on every card after
a short prompt: the experts the answer needs stay on the CPU (it computes 8.0 instead of 1.6 ms of experts a token on
the 16 GB card, 23 instead of 6.6 on the Quadro). After a long prompt it costs less: the cache also moves while a
prompt is read, with it or without it. On the Quadro no setting is faster than none beyond the run-to-run spread,
which is wide there (none: 29.5-32.2 tokens/s after 302 tokens, 19.1-24.3 after 8,192).

How much each one changes the answers, written after the greedy answer without any of them to four prompts (302,
112, 2,048 tokens and a short chat; 896 steps). "Again" is a second and a third run without them, the spread of
two runs of the same settings:

| Card, file | Settings | Same most likely token | Mean KL of the top 20 |
| --- | --- | ---: | ---: |
| RTX 4060 Ti 16 GB, Threadripper 2950X, IQ4_NL (2,619 experts on the card) | again | 98.2%, 99.0% | 0.002, 0.002 |
| | `--expert-swaps 0` | 98.1% | 0.011 |
| | `--loop-bf16` | 98.1% | 0.004 |
| | `--attn-requant q5_0` | 95.6% | 0.032 |
| | `--tail-skip 4` | 97.9% | 0.025 |
| | all four | 95.0% | 0.022 |
| RTX 2080 Ti 11 GB, Ryzen 9 3900X, IQ4_NL (1,582 experts on the card) | again | 98.4%, 98.9% | 0.001, 0.002 |
| | `--expert-swaps 0` | 98.9% | 0.002 |
| | `--loop-bf16` | 98.4% | 0.002 |
| | `--attn-requant q5_0` | 97.0% | 0.014 |
| | `--tail-skip 4` | 96.4% | 0.010 |
| | all four | 94.3% | 0.034 |
| Both cards, UD-Q2_K_XL (the whole file on the card) | `--attn-requant q5_0` | 94.6%, 97.0% | 0.014, 0.023 |

`--loop-bf16` and `--expert-swaps 0` give 98.1-98.9%, about as close as two default runs (98.2-99.0%).
`--attn-requant q5_0` moves the answers most of the three rounding settings. With the whole file on the card,
`--tail-skip 4`, `--expert-swaps 0` and `--loop-bf16` change nothing (100%, KL 0). On the 16 GB card one step of most
settings has a KL of 6 to 8, which sets the means there; so does the default with the doorbell off
(`STRATA_POLY_BELL=0`; the doorbell is how the card hands its misses to the CPU one token at a time). The medians
are 0.0002 and 0.0001 (again), 0.0004 (`--expert-swaps 0`), 0.0003 (`--loop-bf16`), 0.0017 (`--attn-requant
q5_0`), 0.0007 (`--tail-skip 4`) and 0.0025 (all four).

`--tail-skip 4` leaves out more the fewer experts the card holds. With it, the engine's "experts from VRAM" count
also takes the left-out experts as found in VRAM, so it reads high. The engine counts the left-out experts only with
the doorbell off. There the VRAM figure reads high too, since the left-out experts drop out of the count it divides
by. Counted that way, of a token's 384 expert picks (8 in each of 48 layer runs):

| Card (IQ4_NL unless named) | `--tail-skip 4` alone | All four (no swaps, so more misses) |
| --- | ---: | ---: |
| RTX 4060 Ti 16 GB | 3.3 left out (0.9%) | 10.4 (2.7%) |
| Quadro RTX 4000 8 GB | 30.8 (8.0%) | 94.1 (24.5%) |
| RTX 2070 Super 8 GB, UD-Q2_K_XL | 9.4 (2.4%) | 35.8 (9.3%) |
| RTX 2070 Super 8 GB, IQ4_NL (4 cores) | not run | 101.7 (26.5%) |

The answers above were measured on the 16 GB card, where it leaves out 0.9%, and on the 11 GB one, where it was not
counted. On an 8 GB card, where it leaves out about a quarter (24.5-26.5%) of the picks with the other three
settings, they were not measured.

A prompt is read a group of calls at a time through each layer run, so the experts that are not on the card cross
the bus once for the group instead of once for every call and every pass of the looped layers.

`--kv fp8` keeps the attention caches in 8 bits instead of 16, which leaves about 0.7 GB more VRAM for the experts at
a 32K context.
`--kv fp8` needs an NVIDIA card from Ampere on. `--kv q8_0` saves about the same (Q8_0 blocks: 8-bit codes and a
16-bit scale for every 32 values) without FP8 hardware: checked on Turing by its kernel test; AMD not run yet, and a
Mac refuses it. Both are opt-in and change the answers slightly.
With `--kv q8_0`, `STRATA_KV_INT8=1` (the name of Kolibri's opt-in) takes the attention scores of a written token that
reads 32,768 keys or more (only the full-attention layers' tokens do) by `__dp4a`, with the queries quantized to Q8_0
codes too. Written tokens are a token alone (the prompt's last token is read so too) or a window of guessed ones; the
rest of the prompt keeps the float readers, also a chunk of it shorter than 32 rows. The softmax, the values, window
layers and shorter contexts are left as they are. On an RTX 2080 Ti one such layer's attention of a token at 131K went
from 1.76 to 1.57 ms; with the IQ4_NL file, a 131K context and the experts held fixed, KL against the default was
about 1e-6 a token at prompts of 40K to 131K. Off by default: it changes the arithmetic.

On a Mac it runs on strata-poly's Metal engine, as a beta. Measured on an Apple M5 Ultra with 256 GB of unified memory,
UD-Q3_K_XL, the README's benchmark on setup's settings with a 40,960-token context: 256 tokens written, the most likely
one each time. Other programs were using the GPU at the same time. Two runs of the same prompts given to the engine
alone wrote 73-74 and 60-61 tokens/s and read 2,411 tokens/s.

| UD-Q3_K_XL | Writes in a short chat | Reads a 32,787-token prompt | Writes after it |
| --- | ---: | ---: | ---: |
| M5 Ultra | 74 tokens/s | 2,440 tokens/s | 61 tokens/s |

Given the 128 tokens the CUDA engine wrote after a 4,526-token prompt (UD-Q3_K_XL), it picks the same most likely token
at 126 of the 128 steps. UD-Q2_K_XL has run there too (73 tokens/s in the short chat). UD-Q4_K_XL has not been measured
on a Mac. Pictures are read on the Mac's CPU, because the picture tower has no Metal kernels: a screenshot read as 984
rows took 3.4 s.

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** card with **8 GB of VRAM or more**. It has run on an RTX 5090 (32 GB) and on an RTX 4070 Laptop (8 GB). About 3.7 GB are taken before any expert at a 32K context. Or an **Apple Silicon Mac** (beta): it has run on an M5 Ultra. |
| **RAM** | About **21, 17 or 13 GB** beside an 8 GB card for UD-Q4_K_XL, UD-Q3_K_XL and UD-Q2_K_XL, the installer's estimate. A 32 GB card holds any of them whole. On a Mac, setup takes UD-Q3_K_XL from 32 GB of unified memory and UD-Q2_K_XL below that. |
| **Disk** | 9.1 to 17.6 GB for the model file. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver; or macOS 15 or newer. |

**AMD (HIP), experimental, not ready:** the engine builds for RDNA3 and RDNA4 (gfx1100, gfx1200, gfx1201). It has run
on an RX 7900 XT (gfx1100) and an RX 9070 XT (gfx1201), ROCm 7.2.4, in containers that reach the card through
librocdxg: the HIP tests pass (16 of 16) and so does the Inkling 2 kernel test. Two failures of the first runs are
fixed: the first prompt never finished on any AMD card, and with some experts in RAM the RX 9070 XT crashed. With
every expert on the CPU the answers match an RTX 2070 Super's: on the RX 7900 XT the same most likely token at 61 to
64 of 64 steps after a 302-token prompt, and at 15 to 16 of 16 after 2,048 tokens. On RDNA4 the prompt's FP16 products
go through hipBLASLt, which rounds them differently: after one 2,048-token prompt that is unusually sensitive, the RX
9070 XT's answers differed from CUDA's more than the RX 7900 XT's, and with `ROCBLAS_USE_HIPBLASLT=0` as much as the
RX 7900 XT's; after 302 and 8,192 tokens they match CUDA's. Use an NVIDIA card or a Mac for now.

Not run yet: an AMD card on native Linux or Windows, `--expert-gpu` on two cards (built but not run), `--layer-split`
on two unequal cards (below; it has run on two RTX 2070 Super), UD-Q4_K_XL on the 8 GB card and UD-Q4_K_XL on a Mac.
Pictures go through Unsloth's mmproj-F16.gguf, read on the CPU (`--vision cpu`) or the card; on a Mac, on the CPU. The
draft file is an option (`--draft-model`). Several requests at once (`"parallel"`, see [BATCHING.md](../BATCHING.md))
are opt-in on an NVIDIA card (up to 16 slots), not with the draft file, not with `--layer-split` and not on AMD. With
`--tail-skip` (setup's small-card flags) a chat's answer can then depend on the other chats writing at the same time.
On a Mac the engine takes one request at a time for this model, and the installer writes `"parallel": 1` there. Its
license is Apache 2.0.

## Install

```sh
./setup.sh --family inkling
```

Add `--model UD-Q3_K_XL` or `--model UD-Q2_K_XL` for a smaller size.

## Which model should I pick?

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | 4-bit experts, 8-bit where they write ([Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)) | 17.6 GB | about 21 GB |
| **UD-Q3_K_XL** | 3- and 4-bit experts mixed | 13.6 GB | about 17 GB |
| **UD-Q2_K_XL** | 1- to 2-bit experts: the smallest | 9.1 GB | about 13 GB |

The installer downloads the files under the repository's `Mainline_Support` folder, the layout llama.cpp runs
(`inkling2_loop`). The files at the top of the repository (architecture `inkling2`, for example UD-IQ3_XXS) are
Unsloth's layout of the same weights: the engine reads them too. Setup takes them with `--gguf-dir` or
`--custom-gguf`; the engine run by hand takes the file as its `--model`.
`setup --custom-gguf <file>` also takes a quantization the installer does not offer (for example IQ4_NL) as a
size of its own: its experts' size and the RAM estimate are read from the file's header, and the VRAM before any
expert is the closest offered size's. `--custom-gguf` is not used on a Mac yet.

## Two or more cards

`--layer-split` puts the first layers on one card and the rest on the next, as for the other strata-poly models.
Each card keeps its layers' weights, attention caches and expert cache. The loop's twelve layers (6 to 17, which
run three times a token) stay on one card, so the second card starts at layer 1 to 6 or 18 to 22; a number inside
the loop is refused. `"auto"` shares the layers by each card's free VRAM and takes 6 or 18, whichever is nearer.
On two cards whose memory bandwidth differs by more than 5%, `"auto"` also estimates a token's time for every
place to start the second card and both card orders, counting the loop's layers three times, and takes the fastest
when it is at least 3% faster: the loop then goes to the faster card, whatever order the cards are listed in. The
log line `layer split auto: the loop ... on the faster card` says when it did. Equal cards get the plain layout.
`STRATA_POLY_CARD_GBS=a,b` gives the cards' bandwidths in GB/s where the driver's figure is wrong (on AMD cards it
is not read, so they get the plain layout unless it is set). This has not been run on two unequal cards yet. The
idea of keeping the loop on the faster card comes from cobra91 and kuraneko1 (the Strata project's PR #6506).
On two RTX 2070 Super 8 GB (IQ4_NL, a 302-token prompt, 256 tokens written, mean of 3 runs) one card wrote 37.0
tok/s, `--layer-split 18` 51.0 and `6` 48.5. Every split (3, 6, 18, 20, `6,18`, `"auto"`) gave one card's tokens and
top-20 probabilities with every expert on the CPU.
Once a token, a card hands the next one the token's residual streams and the loop's inputs, 64.5 KB. The DSpark
drafter (`--draft-model`) runs on one card only.

## How does it work?

A reply is a list of typed blocks: thinking, text, or a tool call with its arguments as JSON. The server turns
them into the same reasoning, content and tool calls as for the other models, so the OpenAI and Anthropic APIs work
as they do elsewhere. A request that asks for no thinking has its prompt end inside the text block, because the
model otherwise opens a thinking block now and then. More in
[BACKENDS.md](../BACKENDS.md#inkling-2-mini-one-card).

## Checking a change to the engine

Two tools, built with `strata-poly` (CUDA and HIP), check that a change to Inkling 2's code keeps its output:

- `poly_inkling2_test` runs every kernel in `inkling2.cu` on random inputs with the model's shapes and compares them
  with the same math in double precision on the host. Where a fused launch is meant to give the same bits as the
  calls it replaces, it compares the two byte for byte. `--list` names the cases; name some to run only them. It exits
  0 when every check passes, 1 when one fails or none ran, 2 for an unknown case name and 77 with no card (ctest
  skips it).
- `poly_logits_dump MODEL.gguf --tokens @ids.txt --decode @dec.txt --vocab 200058 --out DIR` writes the logits of the
  prompt's last token to `DIR/prompt_logits.f32` and a row for each `--decode` token to `DIR/decode_logits.f32`
  (raw floats, `--vocab` scores a row). Run it with no arguments for the other options (`--greedy N`, `--kv`,
  `--no-graphs`, ...).

Which experts sit on the card changes which code computes them, so pin them for a comparison:
`STRATA_GPU_EXPERTS=N` (a test setting) caches exactly the first N experts of the cache's order and keeps that set
for the whole run (no expert is moved in or out). It is refused with `--layer-split` and `--expert-gpu`, and when
the card cannot hold the N experts. With it the load prints
`strata-poly: cache set N experts fnv HASH`, a hash of the cached (layer, expert) pairs; `N = 0` puts every expert on
the CPU and prints `cache set 0 experts fnv cbf29ce484222325`.

```
STRATA_GPU_EXPERTS=600 poly_logits_dump Inkling-2-Mini-IQ4_NL.gguf --tokens @ids.txt --decode @dec.txt \
    --max-context 4096 --vocab 200058 --out before      # then the same with the other build, --out after
cmp before/decode_logits.f32 after/decode_logits.f32    # and the two "cache set" lines
```
