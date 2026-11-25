# The ZHA index for Inkling 2 Mini (`STRATA_INK_ZHA`)

Back to [Inkling 2 Mini](models/inkling.md)

Inkling 2 Mini's 48 layer runs are 40 that look back 1,024 tokens and 8 that look at the whole context. Those 8
keep 32,768 bytes a token in VRAM (fp16), so a 1M-token context needs 32 GiB for them alone, 17 GiB in Q8_0. With
`STRATA_INK_ZHA` set to a reference file, those 8 runs attend through an index instead: their keys and values go to
pinned RAM in 4 bits, and VRAM keeps a summary of every 64 keys. On an 8 GB card that is what lets a 1M-token prompt
be read and answered.

It is opt-in. Without `STRATA_INK_ZHA` nothing changes: the default path writes the same tokens as before (below,
how that was checked). It runs on an NVIDIA card from Turing on (compute capability 7.5), on one card and one
sequence at a time: with `--batch` above 1 the engine reads one sequence (as with `--kv q8_0`), and the load stops
with a message with `--draft-model`, `--layer-split`, `--expert-gpu`, on AMD and on a Mac.

The method is ZHA ("Zyphra Hierarchical Attention", [arXiv 2610.42112](https://arxiv.org/abs/2610.42112)), one level
of it, with the paper's reference queries trained for this model.

## Turning it on

```
STRATA_INK_ZHA=terminal/data/zha/inkling-2-mini-refs-c64-r4.bin strata-poly --model Inkling-2-Mini-IQ4_NL.gguf --max-context 1048576 ...
```

The reference file is `terminal/data/zha/inkling-2-mini-refs-c64-r4.bin` (131,152 bytes): 4 reference queries for
each KV head of each of the 8 runs, started from k-means of the model's own queries and trained with the paper's
local loss on two long texts. The load prints

```
strata-poly: ZHA index: chunks of 64, 4 references, budget 0.160 (at most 256), a pool of 600 chunks a KV head
```

Three settings tune it; the defaults are the ones measured below.

| Setting | Default | What it does |
| --- | --- | --- |
| `STRATA_INK_ZHA_BUDGET` | 0.16 | the share of a token's far chunks it reads exactly |
| `STRATA_INK_ZHA_MAX` | 256 | at most this many far chunks a token |
| `STRATA_INK_ZHA_POOL` | 600 | an answer's chunks kept in VRAM, a KV head and a layer run; 0 keeps none |

## What a token reads

The keys of a full layer run are cut into chunks of 64. When a chunk is complete, the index summarizes it for each of
4 reference queries of the run's KV head: the keys and the values weighted as that query would weigh them, and the
log of the sum of the weights. A token then reads, for each KV group of 3 heads:

- exactly: chunk 0, its last 1,024 keys, and the far chunks its summaries score best, 16% of them and at most 256
  (the three heads of a group choose together; a prompt's tokens choose 64 at a time);
- through the summaries: every other far chunk as one key, the summary of the reference that fits it best.

The exact chunks come from the 4-bit leaves in pinned RAM over the bus. An answer keeps the chunks it read in a pool
in VRAM (600 a KV head), so the next token reads only the ones it has not read yet. The recent keys stay in a ring
in VRAM in Q8_0. A jump in the sequence (a new prompt, a restored conversation) rebuilds the ring and the summaries
from the leaves.

## What it takes

Per token, for the 8 full layer runs (8 KV heads of 128):

| | a token | at 1,048,576 tokens |
| --- | ---: | ---: |
| the leaves, pinned RAM (4-bit keys and values, fp16 scales) | 9,216 B | 9.0 GiB |
| the summaries, VRAM (4 references a chunk, 4-bit) | 592 B | 592 MiB |
| without the index: fp16 / Q8_0 caches, VRAM | 32,768 / 17,408 B | 32 / 17 GiB |

Beside those the index keeps the answer's pool (338 MiB at the default 600), the ring (7 MiB a layer run at a
2,048-token prompt batch) and, while a prompt is read, work buffers it borrows from the expert cache and gives back
at the end of the read. The summaries' room grows with the context, also from the expert cache.

## Measured

| | |
| --- | --- |
| Graphics card | NVIDIA RTX 4060, 8 GB (sm_89), driver 615.78 |
| CPU and RAM | Ryzen 7 5700X (8 cores), 31 GB |
| Built with | CUDA 13.4, Linux |
| Model | `Inkling-2-Mini-IQ4_NL.gguf` (ggml-org) |
| Command | `strata-poly --model ... --tokens @prompt --max-new N --max-context C --prompt-batch 2048 --kv fp8`, with and without `STRATA_INK_ZHA` |

Reading a prompt and answering. The long prompts are one text with a code hidden at 10% of its length and a question
about it at the end; the answer must be the code (`7349-AMBER-2291`).

| Prompt | Without the index | With it |
| --- | --- | --- |
| 32,768 tokens, 200 written | 1,319 tokens/s read, 31.1 written/s | 1,391 tokens/s read, 32.7 written/s |
| 131,497 tokens, the code | 947 tokens/s, found | 1,328 tokens/s, found |
| 129,024 tokens, 200 written | 963 tokens/s read, 22.2 written/s | 1,310 tokens/s read, 30.0 written/s |
| 271,910 tokens, the code | stops: `no VRAM for the attention cache` | 1,243 tokens/s, found |
| 1,024,338 tokens, the code | (the full runs' fp8 caches alone: 16 GiB) | 853 tokens/s (20 min), found |

At 1M the index's leaves took 9.0 GiB of pinned RAM beside the 11.5 GiB of experts the card did not hold.

How far the answers move: over the last 2,048 tokens of a 32,768-token text, each scored one at a time after the
ones before (a scoring build, not part of this change), the index's next-token distributions against the full
attention's: mean KL 0.0156 (p99 0.216), the same most likely token at 97.1% of the positions, perplexity 1.5656
against 1.5697.

The default path: without `STRATA_INK_ZHA` this build and main (36e654b1b) wrote the same 200 tokens after the
32,768-token prompt, with `STRATA_FREEZE_EXPERTS=1` so that both keep the same experts on the card (which experts sit
there changes which code computes them). With `--batch 3 --check-batch 3` both wrote, for each of 3 sequences, the
same 64 tokens as that sequence alone.

## Limits

- The answers move: a token's far chunks it does not read exactly are estimated. At 32K tokens the next-token
  distributions differ from the full attention's by a mean KL of 0.0156 (above).
- One sequence (`--batch` is not used); not with the drafter, `--layer-split` or `--expert-gpu`; NVIDIA only.
- The reference file is for Inkling 2 Mini, chunks of 64 and 4 references; the load refuses another shape.
