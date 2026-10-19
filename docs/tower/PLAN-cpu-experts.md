# Plan: faster CPU expert reads on Tower (run as Pass 13, 2026-10-18/19: nothing kept)

**Status:** run in full as README.md Pass 13. A (huge pages) and C (repacked Q4_K) gave nothing in the
microbenchmark, even at 100% 2 MiB pages; C's premise was wrong (a task's 32 rows are already one contiguous read).
B, as upstream PR 36, was 7-14% faster at 2-4 tokens a call and 7% slower at 1, and gave no decode gain in the engine
(-4 to -10% fwd, within 3% rev). Tower runs the rebuild with PR 34 and PR 46 and no CPU-expert change.

Written 2026-10-18, to run later in a fresh session. Read [README.md](README.md) first: hardware, ssh access,
the container, and every pass so far. This plan uses its pass numbers.

## Why

Decode on Tower (Cyber-Tiel-Coder-35B-A3B-MTP UD-Q4_K_XL, container `strata-cybertiel`) is mostly CPU expert reads:
57-66% of a token (Pass 6, Pass 11). Those reads top out at about 38 GB/s for the gate and up rows (36 KiB chunks)
and about 30 GB/s for the down rows (5.5 KiB chunks), whether the RAM runs at 2800 or 3200 MT/s (Pass 8). So the
limit is the access pattern, not RAM speed. Thread count, `--window`, down task size and VRAM cache size are already
ruled out (Passes 5, 6, 8, 11).

How strata-poly reads them today (`src/poly/cpu_experts.cpp`, `cpu_experts()`):
- ggml-cpu's plain `vec_dot`, **one row per call** (`nrc = 1`), for gate, up and down.
- Rows come straight from the GGUF's mapping in the **page cache, in 4 KiB pages**.
- No software prefetch.

llama.cpp does three things here that strata-poly doesn't. None of them has been measured on Tower yet.

| # | Idea | What llama.cpp does | Why it might help here |
|---|---|---|---|
| A | Huge pages | `--no-mmap` and repack both put the weights in anonymous memory, which can get 2 MiB transparent huge pages | The 2600X's L2 TLB holds 1,536 4 KiB entries, about 6 MiB of reach, against about 20 GB of experts, so nearly every scattered row is a page walk. 2 MiB pages cover them with about 10,000 entries' worth of reach |
| B | Software prefetch | not in llama.cpp, but cheap | The hardware prefetcher stops at every 4 KiB page edge. A row of gate/up is 1,152 bytes (2,048 values of Q4_K), and a row of down is about 352 bytes |
| C | Repacked rows (`CPU_REPACK`) | ggml-cpu `repack.cpp`: `block_q4_Kx8` interleaves 8 rows, and `ggml_gemv_q4_K_8x8_q8_K` (AVX2) computes 8 rows per call | One contiguous read for 8 rows, and the activation is loaded once instead of 8 times. Same library strata already links |

## Constraints

- **RAM.** A and C need an anonymous copy of the CPU experts, about 20 GB, in place of the page cache. Tower has
  31 GiB usable, with Plex capped at 4 GB, Deluge at 1 GB and swap on the cache SSD. Read the file with `pread`
  into the copy and drop its cache pages (`posix_fadvise(DONTNEED)`) as you go, or the two copies won't fit. The
  catch: a restart no longer starts warm from the page cache. Measure start-up time against today's.
- **THP on Unraid.** Check `/sys/kernel/mm/transparent_hugepage/enabled` and `defrag`. `madvise` mode is enough
  with `MADV_HUGEPAGE`. Confirm it worked with `AnonHugePages` in `/proc/meminfo` or the process's `smaps_rollup`;
  fragmentation after days of uptime can leave you with 4 KiB pages anyway.
- **Types.** UD quants mix types. List the CPU experts' gate, up and down types from the GGUF first. 352 bytes for
  512 values suggests the down rows are Q5_K. Check whether ggml's repack has an AVX2 kernel for each type in the
  pinned ggml. If down has none, C covers gate and up only (about 75% of the expert bytes).
- **Exactness.** A and B change nothing in the arithmetic. C does: the 8x8 kernel can sum in a different order.
  Check with `src/poly/expert_parity.cpp`, the needle test (`needle.py`, 247K) and a quality check like Pass 12's.
- **The house rules (AGENTS.md).** Opt-in behind an env flag. The default path stays byte-identical. Numbers say
  what they were measured on.
- Only one of `strata`, `strata-cybertiel` or koboldcpp can run at a time. Stop `strata-cybertiel` for the
  microbenchmark so nothing else uses the RAM bandwidth, and start it again afterwards.

## Steps

### 1. Microbenchmark (no engine change, about half a day)

Extend Pass 6's `bw.c`/`bw2.c` (branch `claude/tower-profiling-3ccd6b`, copies in `/data/profile/scripts/`) into a
benchmark that does a decode token's CPU expert work: for each of the 40 MoE layers, 8 random experts, gate and up,
then down, on 12 threads, using ggml's real dot products linked from the container's build. Variants:

1. Today: `vec_dot` with `nrc = 1`, rows from the file's mmap.
2. The same from a `MADV_HUGEPAGE` anonymous copy (A).
3. 1 and 2, each with `_mm_prefetch` of the next 1-4 rows (B).
4. Repacked `block_q4_Kx8` with `ggml_gemv_q4_K_8x8_q8_K`, on the huge page copy (C, which includes A).

Report ms per token and GB/s for each, three runs, with the engine stopped. Write it up as the next pass in
README.md.

**Stop rule:** if no variant saves at least 10% of the CPU expert time, stop here and write that down. That is
about 6% of a decode token, and smaller gains get lost in the engine's other costs (Pass 8 saw a micro-benchmark's
8% vanish in the engine).

### 2. Wire the winner in (opt-in)

Only the best variant from step 1, behind one flag (for example `STRATA_CPU_REPACK=1` or `STRATA_HUGEPAGES=1`):
- Load: build the anonymous copy, in repacked form for C, where the experts are mapped today. Keep the mmap path as
  is when the flag is off.
- `cpu_experts()`: for C, compute 8 rows a call, with task sizes in whole groups of 8.
- Parity: `expert_parity.cpp` against the flag-off build.

### 3. Measure in the engine

Same as Passes 10-12: `multiturn.py` fwd and rev, two runs each side, plus short-prompt decode. Also measure
start-up time from a cold and a warm page cache, RSS, and whether Plex or Deluge get squeezed. Then the 247K needle
and the Pass 12 style quality check.

**Keep it** if decode improves by at least 5% across the turns, the needle passes, and start-up and RAM are
acceptable to the user. Then the container gets the flag, the same way `STRATA_FDEC=1` was added (`recreate.sh`,
and the Unraid template once there is one).

## Not in this plan

- ik_llama.cpp's `iqk_mul_mat` kernels. They're faster still on CPU, but it's a fork with its own types and much
  more to port. Look at it only if C wins big and there is more left.
- "Smart expert reduction" (skipping low-weight experts). It's lossy, and strata already has `--tail-skip`.
- GPU-side kernels for the Gated DeltaNet layers. The GPU's share of a decode token is the smaller one.
