# Run PLAN-cpu-experts.md on Tower, with PR 34 + 46 in the rebuild

## Context

`docs/tower/PLAN-cpu-experts.md` (untracked, written 2026-10-18) proposes three ways to make strata-poly's CPU expert
reads faster: (A) huge pages, (B) software prefetch, and (C) repacked Q4_K rows. On Tower these reads take 57-66% of
a decode token. Any of them needs an engine rebuild. The rebuild should also bring in the two upstream PRs that the
live engine doesn't have:

- **PR 34** `--skip-weight F` (merged upstream, `e8a9342a`)
- **PR 46** `--stage-read-mib N` (merged upstream, `26e4cb5d`)

Both are already on the local `tower` branch (origin/main + PR 50/51 + `prefill`, tip `64f17cc2`, which already
contains all of origin/main as of 2026-10-18). The live image `polystrata:tower` is `FROM strata-cybertiel:pr51`
(engine `67f60c12`), which predates them. Both are opt-in flags, so the rebuild should not change default output.

The user wants this run later, in a fresh session, with time set aside to test it properly. This file is the brief
for that session.

**Found while planning (it changes step 1):** upstream **PR 36** (Xiaomi2008, open, head `078594be`) *is* idea B. It
adds `_mm_prefetch` two rows ahead and raises `kRowsPerDownTask` from 16 to 32 in `src/poly/cpu_experts.cpp`, with
byte-identical output. It was measured at +15% kernel GB/s on an i7-13700K and was neutral in serving on a 5090. Test
its exact change rather than writing our own B. Pass 8 already ruled out down task size alone on Tower, so the
prefetch half is what's new. **PR 35** (pin experts, uploads read from the file view) also touches the expert
storage path. Check that it doesn't conflict with an anonymous copy before step 3.

## Before anything: read

- `docs/tower/README.md`: ssh socket (`/tmp/ur.sock`, root@192.168.1.166; the host has no python3), passes 5-12,
  and the "Next" item 4 build recipe (`pass12/Dockerfile.pr51` builds both engines and writes the Strata source
  hash, so containers don't recompile when they start).
- `docs/tower/PLAN-cpu-experts.md`: the why, the constraints and steps 1-3. This plan carries them out and doesn't
  repeat them.
- Memory `tower-branch-and-live-state`: the live container is **`PolyStrata`** from dockerMan template
  `/boot/config/plugins/dockerMan/templates-user/my-PolyStrata.xml`, image `polystrata:tower`, build dir
  `/mnt/user/appdata/polystrata/tower`. `strata-cybertiel` and `strata` are stopped. PLAN-cpu-experts.md still
  says `strata-cybertiel` / `recreate.sh`; read those as `PolyStrata` / the template.

## Phase 0: new baseline image with PR 34 + 46 (no CPU-expert change)

1. `git fetch origin fork`. If PR 50/51 have merged since, rebase `tower` on origin/main. Otherwise merge any new
   origin/main into `tower`. Record the commit.
2. On Tower: build `strata-cybertiel:tower-<sha>` with the `Dockerfile.pr51` recipe, using `tower`'s
   `CMakeLists.txt`, `cmake/`, `include/` and `src/`. Then rebuild `polystrata:tower` `FROM` it (update the
   Dockerfile's FROM and `TOWER_COMMIT`). Keep `:pr51` as the rollback.
3. Apply in the Unraid UI, or `docker` stop/rm/recreate from the template. Check: healthy, both kernel lines
   (FATTN, FDEC) in the log, about 1,988 experts in VRAM, no recompile on start.
4. Baseline, same as Pass 12's numbers: `multiturn.py fwd 3`. Expect about 62 / 55 / 52 tok/s at 32K / 72K / 103K.
   Flags off, so this should match `:pr51` within noise. If it doesn't, stop and find out why before going on.
5. Optional, short: try `--skip-weight` and `--stage-read-mib` once each, to see if either is worth turning on.
   Each is its own decision. Don't fold them into the CPU-expert numbers.

This image is the "today" that every later number compares against.

## Phase 1: microbenchmark (PLAN step 1, about half a day)

Stop `PolyStrata` so nothing else uses the RAM bandwidth. Start from Pass 6's `bw.c`/`bw2.c` (branch
`claude/tower-profiling-3ccd6b`, `/data/profile/scripts/`), linked against the Phase 0 build's ggml-cpu.

First list the CPU experts' gate, up and down types from the GGUF. Then check that the pinned ggml's `repack.cpp`
has an AVX2 gemv for each of those types. If down has none, C covers gate and up only.

Variants (3 runs each, ms/token and GB/s):

1. Today: `vec_dot` with `nrc=1`, rows from the mmap
2. A: the same rows from a `MADV_HUGEPAGE` anonymous copy (confirm `AnonHugePages` and the THP mode in `/sys/...`)
3. B = **PR 36's change** (prefetch 2 rows ahead, down tasks of 32) on 1, and on 2
4. C: `block_q4_Kx8` + `ggml_gemv_q4_K_8x8_q8_K` on the huge-page copy

**Stop rule:** a variant must save at least 10% of the CPU expert time, or stop and write it up as the next pass.
If only B wins, the cheapest outcome is to cherry-pick PR 36 (no RAM cost) and comment on PR 36 with the Tower
numbers (Ryzen 2600X, DDR4). That AMD result is one PR 36 asked for. Posting the comment needs the user's OK.

## Phase 2: wire the winner in (PLAN step 2)

- For B: cherry-pick PR 36's commit onto `tower`. That's all.
- For A/C: one env flag (`STRATA_HUGEPAGES=1` or `STRATA_CPU_REPACK=1`). The load path builds the anonymous copy
  with `pread` + `posix_fadvise(DONTNEED)` per chunk, so the page cache and the copy are never both full. The mmap
  path stays byte-identical when the flag is off. For C, tasks come in whole groups of 8 rows in `cpu_experts()`
  (`src/poly/cpu_experts.cpp`, the gate/up loop near line 186 and the down loop near line 216).
- New RAM risk: an anonymous copy **can go to swap** (8 GiB swapfile, `swappiness=10`), and a page-cache copy
  can't. Either `mlock` it (the container needs `--ulimit memlock=-1` in the template's ExtraParams) or measure
  that it stays resident. On the plus side, the photo-backup `rsync` can no longer evict it (README, "Swap on the
  cache SSD").
- Parity: `src/poly/expert_parity.cpp` against the flag-off build.
- Build every backend the file touches (CUDA here, and HIP if the change goes upstream).

## Phase 3: measure in the engine (PLAN step 3)

Rebuild the image like Phase 0 and add the flag to the template. Then: `multiturn.py` fwd + rev, 2 runs each side,
short-prompt decode, start-up time with a cold and a warm cache, RSS, Plex/Deluge pressure, the 247K needle
(`needle.py`, `TOKENS=185000`, seed 5) and a Pass 12 style quality check. Keep it at 5% or more decode, needle passes,
and start-up time and RAM the user accepts.

## Wrap-up

- Write the passes up in `docs/tower/README.md`: plain words, measured numbers, what each was measured on.
- Commit `PLAN-cpu-experts.md` (it's untracked now) with a status line saying what was run and the outcome.
- Update memory `tower-branch-and-live-state` with the new image tag and rollback.
- Only with the user's OK: a PR comment on PR 36, or our own upstream PR for A/C (opt-in, one change per PR).

## Verification checklist

- Phase 0: flags-off decode matches `:pr51` within noise, and the 107K recall test passes.
- Phase 1: 3 runs per variant, engine stopped, numbers in README.
- Phase 2: `expert_parity` matches when the flag is off; with C, report the max difference.
- Phase 3: the multiturn pairs, the 247K needle, the quality check, start-up time and RSS all recorded.
  Rollback = template back to `polystrata:tower` built on `:pr51`.
