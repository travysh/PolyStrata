# PolyStrata on Tower

Local notes for running and tuning PolyStrata on Tower (Unraid). Not upstream docs. The first trial (with the
Minecraft add-on app as the client) is written up in `~/projects/minecraft_genai/docs/polystrata/README.md`. This
file covers the Strata-specific work after it.

## Hardware

| | |
|---|---|
| Host | Unraid "Tower", 192.168.1.166, headless, root over ssh |
| CPU | Ryzen 5 2600X: 6 cores / 12 threads, Zen+, AVX2, no AVX-512 |
| RAM | 4 x 8 GB DDR4-2800 (31 GiB usable), no swap. Plex, Deluge and the add-on generator share it |
| GPU | RTX 3070 8 GB (sm_86), PCIe 3.0 x16, driver 595.71.05 |
| Disks | Cache pool: Samsung 850 EVO 500 GB **SATA** SSD (`/mnt/cache`, xfs). Array: spinning HDDs behind parity (`/mnt/diskN`), merged by the shfs FUSE layer at `/mnt/user` |

ssh socket: `ssh -fN -o ControlMaster=yes -o ControlPath=/tmp/ur.sock -o ControlPersist=4h root@192.168.1.166`,
then `ssh -o ControlPath=/tmp/ur.sock root@192.168.1.166 '...'`. The Unraid host has **no python3**, so run Python
inside the container: `docker exec strata sh -c "cd /opt/strata && .venv/bin/python ..."`.

## What is running (2026-10-17)

- Images: `strata:latest` (upstream Dockerfile, `CUDA_ARCHITECTURES=86`) and `strata-tower:latest` (adds
  `strata-poly` + the CPU image encoder; `Dockerfile.tower` is in the minecraft_genai folder above and in
  `/mnt/user/appdata/polystrata/`). Both were built from upstream `59efede` (v0.1.45). This clone is at `d0cd84d`, one
  commit later (a bench-tool change only).
- Container `strata`, Qwen3.6-35B-A3B UD-Q4_K_M (Unsloth's **non-MTP** file, accepted through `.done` hash markers):

  ```bash
  docker create --name strata --gpus all -p 8080:8080 --ulimit memlock=-1 --stop-timeout 60 --restart no \
    -v /mnt/user/appdata/polystrata/data:/data -v /mnt/cache/llm:/gguf \
    -e FAMILY=qwen36 -e MODEL=UD-Q4_K_M -e GGUF_DIR=/gguf -e CONTEXT=65536 -e KV=int8 -e VISION=cpu -e GPU=0 \
    strata-tower:latest
  ```

  The only change from the first trial is **`/mnt/cache/llm:/gguf`** where it used to be `/mnt/user/llm:/gguf`.
- `strata-old`: the original container (on `/mnt/user/llm`), stopped, kept as a rollback.
- The `llm` share is now **cache-only** (secondary storage off, set in the Unraid UI). The GGUF and
  `mmproj-F16.gguf` were copied from `/mnt/disk2/llm` to `/mnt/cache/llm` and checked byte for byte with `cmp`. The
  originals are still on disk2.
- Start config: `/mnt/user/appdata/polystrata/data/config/strata-qwen36-ud-q4_k_m.json` (unchanged, 12 CPU threads
  by default). Engine log inside the container: `/opt/strata/strata-qwen36-ud-q4_k_m.log`.

## Pass 1 results: storage, RAM pressure, threads

Harness: `polystrata.py benchmark --model qwen3.6 --prompt-tokens 2000,8000,24000 --json /data/bench-<name>.json`,
run inside the container against the live server. Each prompt gets 256 tokens of greedy answer. It runs twice and
the second pass is kept. JSONs are in `/mnt/user/appdata/polystrata/data/` (`bench-base-*`, `bench-ssd-t{12,6,8}-*`).
Sweep script: `/mnt/user/appdata/polystrata/bench-tsweep.sh N [N...]` (adds `--threads N` to the config,
restarts, benchmarks twice, then restores the config).

Second pass of each config. Prompts: short = 21 tokens; the long ones are 2,862 / 10,795 / 31,919 tokens.

| Config | Decode tok/s (short / 2.9K / 10.8K / 31.9K) | Prefill tok/s (2.9K / 10.8K / 31.9K) | TTFT s (short / 31.9K) |
|---|---|---|---|
| GGUF on array HDD via `/mnt/user` (baseline) | 2.6 / 4.9 / 4.6 / 5.4 | 24 / 80 / 205 | 55 / 116 |
| GGUF on cache SSD, 12 threads (default) | **58.7 / 62.7 / 64.9 / 55.4** | **742 / 1,265 / 1,402** | **0.4 / 17** |
| SSD, `--threads 6` | 56.4 / 60.1 / 61.3 / 53.6 | 759 / 1,276 / 1,406 | 0.4 / 17 |
| SSD, `--threads 8` | 60.4 / 64.6 / 63.6 / 53.9 | 743 / 1,265 / 1,401 | 0.5 / 17 |

Findings:

1. **Storage was the whole problem.** The share had `shareUseCache="yes"`, so the mover had moved the 22 GB GGUF to
   `/mnt/disk2` (an 8 TB IronWolf). strata-poly reads the experts that are not in VRAM in place, through the page
   cache. Free RAM was about 250 MB, so evicted pages came back as random HDD reads through FUSE: `vmstat` showed
   65-76% iowait at about 40 MB/s, with GPU and CPU under 15% busy. The first trial was fast only because the file
   happened to be in RAM. Load time also fell from 36 s to 4-23 s.
2. **RAM pressure (Plex + add-on generator stopped, Deluge running)** changed nothing while the file was on HDD. Both
   baseline passes were the same. It was not re-tested on the SSD.
3. **Threads: no effect.** 6, 8 and 12 are within run-to-run noise (about ±3 tok/s), and prefill is identical. The
   default of 12 stays. (KoboldCPP's gain from 6 threads does not carry over to this engine.)
4. **Warm-up:** the first SSD pass after a restart decoded at 9 / 26 / 53 / 54 tok/s, warming as the expert cache
   learned. Keep discarding the first pass.
5. **Expert cache:** 2,149 of 10,240 routed experts in VRAM (3.8 GiB), 59-73% hits, GPU 40-90% busy, CPU 25-55%.
   Lookup guesses (`--lookup`) were right 45-60% of the time. There are no MTP guesses: the file has no draft block.

### Is the SATA SSD a bottleneck?

Not once warm. After the SSD passes, `vmstat` showed no block reads and 0% iowait, so the experts came from the page
cache (RAM). Reads during the runs were not logged. The SSD matters at load and on the first pass, and whenever the
page cache gets evicted. Then it is about 500 MB/s sequential, with random reads much slower. The risk is RAM, not the
SSD, and it doesn't show as "used" RAM. `free -g` during the runs showed 8-9 GiB used (the engine process plus
everything else), 25-27 GiB buff/cache, and about 0.25 GiB free, all of it system RAM. The GGUF is not loaded into
the process: it is mmapped and lives in the page cache, which monitors count as cache or "available", not used. Only
the experts in VRAM (3.8 GiB) are copied out. The roughly 80% of experts run on the CPU are read from those cached
pages, so they have to stay cached. When the cache can't hold them next to Deluge's I/O and everything else, pages get
dropped and re-read from disk, which is what happened on the HDD. To check on a run: `vmstat 2` on Tower
during a benchmark. Sustained `bi` and `wa` above a few % mean the model no longer fits in the page cache.

## Next: Cyber-Tiel-Coder 35B-A3B UD-Q4_K_XL

peculiar-ragdoll's abliterated Tiel Coder, built on Huihui-Ornith-1.5-35B-A3B-abliterated: the Qwen3.6 architecture,
with the Sharp chat template in the GGUF. PolyStrata has a `tiel` family (`setup.py`, around line 641) for the
**non-abliterated** Tiel Coder, pinned to `peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP` at `bbe9e566`. It has no
entry for CyberTiel.

### Which file: the MTP repo or the plain one

The 22.7 GB size quoted for this model matches the **MTP** repo's file, not the plain one's:

| Repo (revision) | File | Bytes | SHA-256 |
|---|---|---|---|
| `peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP` (`fa19d4f33561dc0d107c2a2f8943f1ca2e288109`, 2026-09-18) | `Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf` | 22,749,880,640 | `0bbcf3cc9be4c976bad20e641baf629dad9c178d39ebdc9cd72129179943c06a` |
| same | `mmproj-BF16.gguf` | 902,822,016 | `d9ce31026d1cb1f3f8d5152e2e2a014d9d2b302b6c93a7dc07bb0a0487f52837` |
| `peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF` (`644bee2e6b9a75aa64c3bb818aaeeebe43d862bb`, 2026-09-10) | `Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf` | 22,360,478,720 | `21bff5a96fa051809f4a8260befe977771c9927cae9c32eadfcb0718bae960cd` |
| same | `mmproj-BF16.gguf` | 902,822,016 | `d9ce3102...` (identical) |
| same | `mtp-Cyber-Tiel-Coder-35B-A3B.gguf` (the draft block alone) | 1,481,065,248 | `34f22d95765353f538f035119d7a73fda72b3ac1a4dd083e234d5eb10e5ae341` |

- **Decision (2026-10-17): use the MTP repo's file.** It has the draft block inside it (`"draft": "own"`, as the `tiel` family
  expects), so the engine's MTP guesses work. That is the lever the first pass could not test (`draft_n` stayed about
  0). The extra 389 MB is that block. The file is 480 bytes larger than the pinned Tiel MTP Q4_K_XL
  (22,749,880,160), presumably metadata.
- The plain file plus the separate `mtp-...gguf` through `--draft-model` might also work, but it is unverified.
  `Engine::load_shape()` (`src/poly/engine.cpp:887`) needs `blk.<n_layer>.nextn.eh_proj.weight` in the draft file,
  and it drops the block if the file has `nextn.embed_tokens.weight` or `nextn.shared_head_head.weight` (the engine
  doesn't run its own token table). Check the tensor names first with `gguf-py`.
- `mmproj-BF16.gguf` has the **same hash** as the one `tiel` pins, so it passes setup's check as it is, with no
  marker.
- RAM: the `tiel` UD-Q4_K_XL entry says `ram_gb: 27` beside an 8 GB card. The file is about 0.6 GB larger than
  today's Qwen3.6 file, so the page-cache margin gets tighter (see the SSD section above).

### Ways to make PolyStrata take it

1. **Fork and add a `cybertiel` family (recommended; it's what the fork is for).** Copy the `tiel` entry in
   `FAMILIES` with: `hf`/`mmproj_hf` = the Cyber MTP repo, `file` = `Cyber-Tiel-Coder-35B-A3B-MTP-{q}.gguf`, `tag` =
   `cybertiel-`, `name` = `cyber-tiel-coder-35b-a3b`, the sizes wanted (at least UD-Q4_K_XL), and the real
   `sha256`s above. Add the repo and revision to `HF_REVISIONS` (`setup.py:73`). Things to update with it:
   - the `--family` choices and help (`setup.py:7026`)
   - the HIP exclusion tuple (`setup.py:6287`)
   - `tools/strata_mcp.py` (its own copy of the families, around line 156)
   - the menu and model tests (`tools/test_setup_menu.py`, `tools/test_setup_gguf_models.py`), which list the
     families

   Run `python tools/test_setup_*.py`; they need no GPU. Then, rather than a full rebuild (which takes about 22
   minutes and recompiles both engines), add a thin layer, `FROM strata-tower:latest` + `COPY setup.py
   /opt/strata/setup.py`. That's enough because only Python changes. With `FAMILY=cybertiel MODEL=UD-Q4_K_XL`,
   the entrypoint finds no config and runs setup, which writes the tokenizer pack and the config. A later change to
   `src/`, `include/` or `CMakeLists.txt` does need the full rebuild.
2. **No code: reuse the `tiel` family with a hash marker,** as was done for Qwen3.6. Put the file in `/gguf` under
   the name `Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf` and write
   `Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf.done` = `sha256 54f46c4ce544c225122b0f066c2336f10404be7bc53b0cc94d1dbbc5e826bdc1`.
   The size differs from the pin, so without the marker `verify_sha256` fails on size (it does not delete). It works,
   but the server then reports the model as Tiel Coder, and the pack and config are shared with the real Tiel.
3. **Hand-written start config:** copy the current `strata-*.json` and change `--model`, `vision.mmproj` and
   `vision.model`. This skips setup, but the tokenizer pack and expert profile are per model (setup builds the
   tokenizer with `tools/strata_tokenizer.py`), so it needs those steps run by hand.

### What was done (2026-10-17)

- Downloaded the MTP repo's UD-Q4_K_XL and `mmproj-BF16.gguf` (pinned `fa19d4f3`) to `/mnt/cache/llm`. Both
  SHA-256s match the table above.
- Option 1: a `cybertiel` family in this clone (`setup.py`: `HF_REVISIONS`, `FAMILIES`, the HIP tuple and the
  `--family` help; `tools/strata_mcp.py`; `tools/test_setup_menu.py`). UD-Q4_K_XL is its only size. Not committed and
  not forked yet. The setup tests on Linux (inside the image) match the base image: `test_setup_golden` fails the same
  8 `unsloth` cases with or without the change. Five tests stop on the bare image's missing tokenizer pack. On macOS
  many tests fail both ways, because `main()` takes the Mac path.
- Image `strata-cybertiel:latest` = `strata-tower:latest` + `COPY setup.py` + `COPY tools/`
  (`/mnt/user/appdata/polystrata/cybertiel/Dockerfile`). Container `strata-cybertiel`: the same flags as `strata`, with
  `FAMILY=cybertiel MODEL=UD-Q4_K_XL`. Setup checked the hashes, built the pack (`/data/packs/cybertiel-ud-q4_k_xl`),
  wrote `/data/config/strata-cybertiel-ud-q4_k_xl.json` and started it. Loaded in 10.7 s: 2,002 of 10,496 routed
  experts in VRAM (3.6 GiB). `strata` is stopped.

## Pass 2 results: Cyber-Tiel-Coder UD-Q4_K_XL (MTP)

The same harness and settings as pass 1 (12 threads, 64K context, int8 KV, cache SSD). JSONs:
`/data/bench-cybertiel-{1,2}.json`, `vmstat` logs: `/data/vmstat-cybertiel-{1,2}.txt`. Second pass. The short prompt is
201 tokens here (the Sharp template's system text), against 21 for Qwen3.6. The 10.8K and 31.9K answers stopped early
(135 and 179 tokens of 256).

| Config | Decode tok/s (short / 3.0K / 11.0K / 32.1K) | Prefill tok/s (3.0K / 11.0K / 32.1K) | TTFT s (short / 32.1K) | MTP right |
|---|---|---|---|---|
| Qwen3.6 UD-Q4_K_M, no MTP (pass 1, SSD, 12 threads) | 58.7 / 62.7 / 64.9 / 55.4 | 742 / 1,265 / 1,402 | 0.4 / 17 | - |
| Cyber-Tiel UD-Q4_K_XL, MTP | **61.3 / 66.8 / 68.3 / 58.0** | 762 / 1,261 / 1,413 | 1.1 / 17 | **82-93%** |

Findings:

1. **MTP is guessed right 74-93% of the time.** Against Qwen3.6 decode is only 4-6% faster, but that compares
   different weights at different quants. The clean on/off test below puts MTP at +12 to +25%.
2. **Prefill and TTFT are unchanged** (prefill is batched and doesn't use the draft).
3. **Page cache: fine once warm.** Pass 1 read the new file from the SSD: mean 86 MB/s in, peaking at 397 MB/s, iowait
   13% on average and up to 71%, and decode on the short prompt was 40 tok/s. Pass 2: mean 2.6 MB/s in, iowait 0.5%
   (max 4%). `free -g` afterwards: 8 GiB used, 26 GiB buff/cache, 22 GiB available. The 0.6 GB larger file still fits
   beside Deluge, with Plex and the add-on generator stopped.
4. Expert cache: 42-57% hits (Qwen3.6: 59-73%), GPU 38-87% busy, CPU 27-76%.

## Pass 3: MTP on/off on the same file

`strata-poly --no-draft` ([main.cpp:1153](../../src/poly/main.cpp)) does not run the draft block. The engine then
loads 10,240 routed experts instead of 10,496: the block's own 256 are dropped. `--lookup 3` stays on in both
(the default, and how the Qwen3.6 pass ran), so "MTP right" without the draft is the lookup guesses alone. The order
was off, then on, with a restart before each and two passes each, keeping the second. JSONs:
`/data/bench-cybertiel-{nodraft,draft}-{1,2}.json`. `vmstat` in both second passes: under 1 MB/s in, iowait at most 1%.

| Cyber-Tiel UD-Q4_K_XL | Decode tok/s (short / 3.0K / 11.0K / 32.1K) | Prefill tok/s (3.0K / 11.0K / 32.1K) | Guesses right |
|---|---|---|---|
| `--no-draft` (lookup only) | 51.3 / 53.4 / 54.4 / 48.1 | 759 / 1,266 / 1,417 | 33-60% |
| draft on (this session) | 58.4 / 65.9 / 68.0 / 53.7 | 758 / 1,262 / 1,416 | 74-89% |
| draft on (pass 2, an hour earlier) | 61.3 / 66.8 / 68.3 / 58.0 | 762 / 1,261 / 1,413 | 82-93% |
| **draft on vs off** | **+14 / +23 / +25 / +12%** (this session's on) | the same | |

Findings:

1. **MTP is worth +12 to +25% decode on this card**, the most on the 3K and 11K prompts. The two draft-on runs differ
   by up to 4 tok/s (about 7%), so the gain is clearly above the noise, but the exact per-prompt figure is not.
2. **Prefill is the same with and without** (it doesn't use the draft).
3. **Without the draft, Cyber-Tiel Q4_K_XL decodes 6-13% slower than Qwen3.6 Q4_K_M** (51-54 against 59-65 tok/s).
   That fits its larger quant and lower expert-cache hit rate (46-60% against 59-73%). It is a different model, so
   this is not an engine finding.
4. Decode is still far under what an 8 GB card can do with the experts in VRAM. The CPU expert path (Zen+, 12
   threads, about 80% of the experts) is the next lever.

## Pass 4: bigger context, `--kv q8_0`

The container's `KV=int8` never reaches strata-poly. It is the Qwen3.8 engine's setting, setup does not pass
`--kv q8_0` for the poly families, and so every pass above ran fp16 attention caches. Here `--kv q8_0` and/or
`--max-context 262144` (the trained length, no rope scaling) were added to the config by hand, with draft on and
everything else the same. Prompts: short, `--prompt-tokens 24000,80000` = 32,099 and 107,310 tokens. Second pass kept.
JSONs: `/data/bench-cybertiel-{q8-256k,fp16-256k}-{1,2}.json`. Afterwards the config was put back to 64K fp16.

| Cyber-Tiel UD-Q4_K_XL, draft on | Decode tok/s (short / 32K / 107K) | Prefill tok/s (32K / 107K) | TTFT s (32K / 107K) | Expert cache hits (32K / 107K) |
|---|---|---|---|---|
| 64K, fp16 (passes 2 and 3) | 58.4-61.3 / 53.7-58.0 / - | 1,413-1,416 / - | 17 / - | 54% / - |
| 256K, fp16 | 44.2* / 48.2 / 31.6 | 1,585 / 919 | 20 / 86 | 44% / 33% |
| **256K, q8_0** | **59.7 / 54.8 / 35.3** | 1,559 / 904 | 21 / 87 | 52% / 50% |

\* The short prompt runs right after a 107K one, while the expert cache is still shrunk (the log: 559 experts, 1.0
GiB, before it gave back 1.55 GiB). Pass 1's short prompt, after a restart, was 64.4. With q8_0 the dip was smaller
(59.7, pass 1: 63.5).

Findings:

1. **256K context with `--kv q8_0` costs nothing on short and 32K prompts.** Decode is within the run-to-run spread of
   the 64K fp16 config (about ±4 tok/s).
2. **At 107K, q8_0 decodes 12% faster than fp16** (35.3 against 31.6): the cache is half the size, so the expert cache
   keeps more VRAM (50% hits against 33%). fp16 at 256K is also slower at 32K (48 against 55).
3. **Long prompts are prefill-bound:** 107K tokens take about 87 s before the first token (about 900 tok/s, falling
   from about 1,550 at 32K). It is the same with fp16 and q8_0.
4. TTFT at 32K is 20-21 s here against 17 s at 64K. The 64K runs had prompt-cache reuse between passes (the log:
   20,480 of 92,634 tokens not read again), so the TTFTs are not directly comparable. Prefill tok/s is the cleaner
   figure.
5. RAM: about 8 GiB used, iowait at most 4%. The longer context doesn't press on the page cache, because the attention
   caches are in VRAM.
6. **Long-context recall with q8_0: 6 of 6.** `docs/tower/needle.py` (also `/data/needle.py` on Tower) hides three
   access codes at 3%, 50% and 92% depth in the benchmark's manual text, about 107,400 tokens, and asks for all three
   (temperature 0). Two prompts with different codes: all six were returned word for word, 108-110 s per request.
   It is a simple test: unique strings in repeated text. It shows that the q8_0 cache holds facts from early in the
   prompt, not that long-context reasoning is unchanged.

7. **A nearly full context works: no OOM.** One recall prompt of 247,377 tokens (`TOKENS=185000` for
   `needle.py`, 94% of 262,144) returned all 3 codes, in 409 s for the whole request (about 600 tok/s of prefill).
   Sampled every 5 s: GPU memory at most 7,639 of 8,192 MiB. The engine made room by shrinking the expert cache to
   1,159 experts (2.06 GiB, from 3.6 GiB at load), leaving 524 MiB free. System RAM at most 8.3 GiB used, at least
   23.7 GiB available. No OOM lines in `dmesg`, and the container did not restart. Memory log:
   `/data/mem-256k.txt`.

**Tower's config is now 256K + `--kv q8_0`** (2026-10-17). The 64K fp16 original is
`/data/config/strata-cybertiel-ud-q4_k_xl.json.orig`: copy it back and restart to undo. Since Pass 7, the container's
env (`CONTEXT=262144 KV=int8`) gives the same config through setup, so a recreate with `REINSTALL=1` writes it too.

**Hand edits to the config survive a restart (checked 2026-10-18).** setup.py writes the config file again on every
container start, so its time changes each start. The arguments stay as they were, though: a hand-added
`--window 4` was still in the file and on the engine's start line after `docker restart`, and was removed afterwards.
`REINSTALL=1` is different: it builds the config from setup's own choices, so hand edits are lost, and the order can
change (it put `--kv q8_0` at the end). The reference copy `.json.q8-256k` was refreshed from the current config after
the Pass 7 recreate, so `cmp` against it holds again.

## Pass 5: guess length (`--window`)

`--window N` is the most tokens that go through the model per step: the known one plus guesses (default 8). The
draft block guesses at most 3 of them (`kBlockWindow = 4`, `src/poly/main.cpp:68`), and lookup guesses can fill the
rest. The engine already adds a guess only while its own timings say it pays (`guesses.wants`). The Cyber-Tiel card
found 1 draft token fastest in llama.cpp (1.21x, against 0.57x at 8), which is why this was tried. Config: 256K +
`--kv q8_0` + `--window W`, three passes each, decode tok/s as the mean of passes 2 and 3. JSONs:
`/data/bench-cybertiel-w{2,4,8}-{1,2,3}.json`.

| `--window` | short | 3.0K | 11.0K | 32.1K | mean | Guesses right |
|---|---|---|---|---|---|---|
| 2 | 62.5 | 62.4 | 64.8 | 51.0 | 60.2 | 79-88% |
| 4 | 63.1 | 62.5 | 66.0 | 54.7 | 61.6 | 78-91% |
| 8 (default) | 56.3* | 65.9 | 65.0 | 57.8 | 61.3 | 75-90% |

\* One pass was 50.1, the other 62.5.

Findings:

1. **No window is clearly faster.** The means are within 1.5 tok/s, and single cells move by 4-8 tok/s between passes
   (the answers also stop at different lengths). Only `--window 2` at 32K looks lower (51 against 55-58), at the edge
   of the noise. **The default (8) stays**, and the config is unchanged.
2. The llama.cpp result doesn't carry over, as expected. llama.cpp drafts a fixed `n_max`, while strata-poly already
   stops guessing when a guess stops paying, and caps the block at 3.

## Pass 7: a growing coding conversation (follow-up turns)

The setup fix (commit `24aadb56`, local only): `--kv int8` now gives the poly families `--kv q8_0`, and other values
warn. Without `--kv` the config is unchanged. The container was recreated with `CONTEXT=262144 KV=int8` and once with
`REINSTALL=1`: setup alone wrote `--max-context 262144 ... --kv q8_0`, so the container's settings and the engine
now agree. It was then recreated without `REINSTALL`.

`docs/tower/multiturn.py` (also `/data/multiturn.py`) builds a conversation like a coding agent's. Each turn adds
about 30K tokens of PolyStrata's own source and asks about the last file. The answer goes back into the history, and
the conversation grows to about 235K. "Read" is the prompt tokens the engine actually processed. The rest came from
its prompt cache.

**Run A** (with an answer each turn): up to 600 tokens of thinking (`reasoning_budget_tokens`) and 1,200 in all,
temperature 0. Every turn ended with an answer (`finish_reason: stop`, 1,250-1,900 characters). Plex and the add-on
generator were running.

| Turn | Prompt tokens | Read | Prompt time s | Read tok/s | Out | Decode tok/s | Expert-cache hits |
|---|---|---|---|---|---|---|---|
| 1 | 31,647 | 31,647 | 20.5 | 1,544 | 987 | 53.5 | 52% |
| 2 | 72,308 | 40,662 | 41.4 | 982 | 1,071 | 39.8 | 41% |
| 3 | 102,836 | 30,529 | 40.9 | 747 | 1,076 | 35.2 | 38% |
| 4 | 142,606 | 39,771 | 67.9 | 586 | 991 | 29.3 | 31% |
| 5 | 173,997 | 31,392 | 66.3 | 473 | 1,011 | 25.6 | 27% |
| 6 | 205,233 | 61,873 | 130.6 | 474 | 667 | 22.6 | 21% |
| 7 | 236,258 | 31,026 | 84.5 | 367 | 1,067 | 20.8 | 12% |

**Earlier runs B and C** (`max_tokens` 400 and no budget): every reply ran out while still thinking, so the history
got empty answers. Their decode and read figures match run A's within a few tok/s at each fill:

| Run B (forward) turn: prompt, read, read tok/s, decode, hits | Run C (reversed) turn: prompt, read, read tok/s, decode, hits |
|---|---|
| 1: 31,647, 31,647, 1,105, 24.3, 60% | 1: 35,229, 35,229, 1,536, 44.5, 40% |
| 2: 71,931, 40,289, 972, 35.5, 40% | 2: 77,200, 41,976, 977, 41.1, 31% |
| 3: 101,998, 30,072, 761, 34.4, 35% | 3: 123,269, 46,074, 696, 36.2, 33% |
| 4: 141,302, 39,309, 610, 28.7, 29% | 4: 203,727, 80,463, 490, 25.1, 19% |
| 5: 172,312, 31,015, 500, 28.1, 28% | 5: 238,982, 35,260, 384, 20.4, 10% |
| 6: 203,147, 30,840, 438, 24.7, 23% | |
| 7: 233,865, 30,723, 386, 22.1, 15% | |

Findings:

1. **The prompt cache works on follow-up turns.** A turn reads about what is new: the added code and the last answer.
   The one exception is run A's turn 6, which read 61,873 tokens for about 31K new, so about 31K were read again. The
   cause wasn't looked into. The history drops the thinking the engine had in its cache, so the prompt stops matching
   at the last answer, and the engine goes back to a checkpoint. Usually that checkpoint is close, but here it wasn't.
2. **Each follow-up still gets slower as the context fills.** Reading new tokens slowed from about 1,100-1,550 tok/s
   near 30K to about 370 tok/s past 230K, because each new token's attention reads the whole history. A 30K-token turn
   takes about 20-29 s to read near the start and about 85 s past 230K.
3. **Decode falls with the fill:** about 40-54 tok/s up to 75K, about 35 at 100-125K, 26-29 at 140-175K, 21-25 past
   200K, in all three runs. Expert-cache hits fall from 40-60% to 10-15% as the attention cache takes the VRAM (the
   log: the cache was down to 312 experts at one point, then got 2.23 GiB back).
4. **Code against the benchmark's text: no clear difference.** At about 32K, code decoded at 24.3, 44.5 and 53.5 tok/s
   in the three runs, and the benchmark's 32K prompt at about 55. The spread between runs is too wide to call it.
5. **RAM with Plex and the add-on generator running:** runs B and C read 31 MB/s from disk on average, with iowait 2%
   on average and peaks of 63%. Run A, an hour later, read 0.1 MB/s with iowait at most 8%. Available RAM was 23.6-25.2
   GiB. The evictions come and go, presumably with what Plex and the add-on generator are doing. This was with the
   Plex (4 GB) and Deluge (1 GB) memory caps from "Shared-load changes" below in place.

**For coding:** 256K stays the server's limit, so a long one-off job still fits. A coding agent is best set to
compact or summarise its history at about **100-130K tokens**. Below that, answers decode at about 35 tok/s or more
and a 30K-token turn reads in about 40 s or less. Past it, both get steadily worse. Give requests a thinking budget
(`reasoning_budget_tokens`, per request or in the config for all requests). At `max_tokens` alone, a reply can end
while still thinking and come back with no answer.

## Pass 8: RAM speed (2400 / 2800 / 3200 MT/s)

The BIOS fell back to 2400 MT/s after a reboot (probably a failed memory training at 2800: four DIMMs from two
Ballistix 3000 CL15 kits on Zen+). It was then set to 3200 MT/s at 1.36 V. (SMBIOS reports 1.2 V; this board reports
the default there.) The same sequence after each boot: the 107K recall prompt as a warm-up (it passed every time),
then `multiturn.py fwd 5` with the thinking budget. The 2800 column is Pass 7's run A, which was not right after a
boot. Plex and the add-on generator ran in all three. Decode tok/s (read tok/s):

| Turn, context | 2400 | 2800 | 3200 |
|---|---|---|---|
| 1, 32K | 46.2 (1,411) | 53.5 (1,544) | 51.5 (1,540) |
| 2, 72K | 37.6 (1,020) | 39.8 (982) | 40.2 (988) |
| 3, 103K | 33.1 (747) | 35.2 (747) | 35.1 (761) |
| 4, 143K | 28.6 (609) | 29.3 (586) | 29.1 (592) |
| 5, 174K | 25.4 (493) | 25.6 (473) | 26.3 (497) |

Findings:

1. **3200 is the same as 2800, within noise** (decode within about 1 tok/s from turn 2 on, and reading the same). 2400
   is 1-6% slower on decode from turn 2 on, with the same reading speed. Turn 1 varies most between runs.
2. **Why the RAM speed doesn't show:** the expert reads don't get faster with it. The profiling session's micro-benchmarks
   (Pass 6, on branch `claude/tower-profiling-3ccd6b`: `/data/profile/scripts/bw.c`, `bw2.c`, built in the container
   with its gcc 13), run again at 3200 with the engine idle, against Pass 6's figures at 2800 (GB/s):

   | Read pattern | 2800 | 3200 |
   |---|---|---|
   | sequential, 6 threads / best (3 threads) | 40.1 / 40.1 | 43.7 / 45.6 |
   | random 36 KiB chunks (32 rows of gate or up), 12 / 6 threads | 37.9 / 38.0 | 38.5 / 41.6 |
   | random 5.5 KiB chunks (16 rows of down), 12 / 6 threads | 29.9 / 30.2 | 30.2 / 31.2 |

   Sequential reads gained 9-14% from 2800 to 3200, but the expert-sized random reads at the engine's 12 threads
   gained 1-2%. So the CPU side is held back by the reads' size and scatter (each chunk is a fresh DRAM access the
   prefetchers can't run ahead of, with 12 threads contending), not by the RAM's transfer rate. This corrects an
   earlier line here that put it down to synchronisation: Pass 6 measured synchronisation at 8-9% of a token. Two
   things follow. Larger down tasks (Pass 6 candidate 3: 64 rows, 22 KiB) matter more than RAM speed. And at 3200, 6
   threads read the 36 KiB chunks 8% faster than 12 (they were equal at 2800), so `--threads 6` is worth testing again.
3. Stability at 3200 during this run: no new kernel errors, no container restarts, GPU at most 66 C. That is about 10
   minutes of load, not a memory test. MemTest86 is still to be run.
4. **`--threads 6` against 12 at 3200: 12 stays.** Same sequence (restart, the 107K warm-up, `multiturn.py fwd 5`), 6
   first, then 12. Decode tok/s for turns 1-5 (32K to 174K), 6 threads: 47.6 / 38.5 / 33.9 / 29.1 / 24.7. 12 threads:
   52.4 / 41.5 / 34.5 / 30.5 / 25.8. 12 was faster at every turn, by 2-9%. Reading was the same (within 6%). The
   micro-benchmark's gain for 6 threads on 36 KiB reads doesn't carry over to the engine, where the threads also do
   the dot-product arithmetic and the 5.5 KiB reads.

## Pass 9: prompt reading at long context (interim, 2026-10-18)

At long context a coding turn waits mostly for its prompt to be read: a ~35K-token follow-up takes ~65 s to read at
140-175K and its ~1,000-token answer ~30 s. All runs below: RAM 3200 MT/s, 256K + `--kv q8_0`, 12 threads, Plex
and the add-on generator running, the Pass 8 sequence (restart, the 107K recall warm-up, `multiturn.py fwd 5`).
Read tok/s per turn (context 32K / 72K / 103K / 143K / 174K):

| Config | Read tok/s | Decode tok/s | Kept? |
|---|---|---|---|
| Baseline (Pass 8 and the threads-12 run) | 1,530-1,540 / 988-1,026 / 749-761 / 587-592 / 495-497 | 51.5-52.4 / 40.2-41.5 / 34.5-35.1 / 29.1-30.5 / 25.8-26.3 | |
| `--prompt-batch 2048` | 1,586 / 1,094 / 752 / 610 / 511 | 51.2 / 39.5 / 34.4 / 29.1 / 25.2 | no: +0-4% read at long context, -1-3% decode (0.4 GiB less expert cache: 1,748 experts) |
| Measuring build, no change (control) | 1,437 / 1,022 / 749 / 610 / 496 | 52.3 / 41.5 / 35.1 / 30.6 / 26.7 | |
| `STRATA_ATTEND_MIB=16` (1,024 queries a chunk, not 128) | 1,537 / 1,033 / 728 / 619 / 498 | 52.8 / 39.6 / 35.2 / 28.9 / 26.9 | no: noise |
| + `STRATA_SCORE_MIB=256` (key blocks back to 2,048) | 1,549 / 1,050 / 769 / 625 / 491 | 50.9 / 40.3 / 33.8 / 29.5 / 26.4 | no: noise (and -105 experts) |
| `"preserve_thinking": true` (config) | 1,538 / 1,004 / 742 / 577 / 466 | 51.0 / 40.0 / 34.6 / 29.8 / 26.5 | no: each turn read ~350-450 tokens less, but the kept thinking grew the context; turn time same or 1-3 s worse |

Where the time goes (from the numbers, not yet profiled): a 1,024-token batch takes about 0.36 s plus 9.8 ms per 1K
tokens of context, so at 140K about 80% of a batch grows with the context. That part is attention: 10 of the 40
layers, 16 heads of 256 (2 KV heads), about 168 GFLOP per batch per 1K of context, i.e. it runs at about 17 TFLOPS
against about 40 for the 3070's tensor cores (FP16 in, FP32 sums). The prompt path, `attend_batch()` in
`src/poly/kernels.cu`, is not fused: per block of keys (2,048 under q8_0, `kQ8WorkRows`) it dequantizes K and V,
writes fp32 scores with cuBLAS, runs a softmax kernel over them, and multiplies by V with cuBLAS, 128 queries at a
time (`kAttendResults`). Chunk sizes made no difference (above), so the lever left is a fused flash-attention prompt
kernel. Estimate, if it reaches 60-70% of the tensor-core peak: +15% read at 32K, +29-44% at 140K, +33-50% at 230K.

Measuring build: image `strata-cybertiel:p9` on Tower (`/mnt/user/appdata/polystrata/pass9/`: the patched
`kernels.cu` / `qwen35.cpp` with the two env variables, `Dockerfile`, `p9run.sh` for A/B in a container `strata-p9`,
`nsys.sh` + `cap.py` for the profile, not run yet). Results: `/data/pass9-*.txt`.

### Next

1. Upstream: offer `24aadb56` (`--kv int8` gives the poly families q8_0), with the numbers above. Kept local for now.
2. The CPU-path levers: a smaller quant (check whether the Cyber repo has UD-Q3_K_XL / UD-IQ3_XXS like Tiel's), or
   profiling the CPU expert path on Zen+.
3. Fork and push the `cybertiel-family` branch (the family commit only; these notes stay local).

## Pass 10: flash attention for prompt reading (2026-10-18)

**Profile.** Nsight Systems 2026.3.2 (`nsys launch` in a container `strata-nsys`, CUDA trace only), the server on its
normal config with `strata-cybertiel` stopped. A ~138K-token conversation was filled first (137,863 tokens read in
153.7 s, not captured). Then two captures: a ~4K-token follow-up read at ~138K of context (4,082 tokens read in
9.9 s), and the same ~4K tokens as a conversation of their own, at ~0 of context (4,258 tokens in 4.9 s). Each
reply was 8 tokens with thinking off, so the captures are almost all prompt reading. GPU time per kernel, ms:

| Kernel | ~138K | ~0 | Grows by |
|---|---|---|---|
| cuBLAS PV product (`cutlass_80_tensorop_s16816gemm_f16_256x128`, "nn") | 1,925 | 32 | 1,894 |
| `attn_softmax_block_kernel` | 1,738 | 28 | 1,710 |
| cuBLAS QK product (`ampere_s1688gemm_fp16_256x64`, "tn") | 1,716 | 31 | 1,685 |
| `dequant_q8_kernel` | 166 | 3 | 162 |
| `attn_scale_kernel` | 147 | 7 | 140 |
| the decode attention of the 8 answer tokens (`attn_score_reg`, `attn_softmax`, `attn_value*`) | 78 | 2 | 76 |
| MMQ (`mul_mat_q`, experts and projections) | 878 | 862 | 15 |
| `gdn_state_reg_kernel` (the 30 linear-attention layers) | 342 | 356 | -15 |
| `gemm_small_kernel` | 203 | 210 | -6 |
| everything else | 313 | 316 | -3 |
| **all kernels** | **7,506** | **1,847** | **5,660** |

Which product is QK and which is PV is read from their transpose flags (`attend_batch()` calls QK as T,N and PV as
N,N). So the prompt attention (the two products, the softmax, the scaling and the dequantization) grows by 5,590 ms
from ~0 to ~138K, which is all of the growth. At ~138K it is 5.7 s of the 7.5 s of GPU time, in a 9.9 s read.
The softmax kernel takes as long as either product: it reads and writes the fp32 scores the QK product left in
memory. The products themselves ran at about 26 TFLOPS. A fused kernel removes the softmax's memory traffic.

**The port.** `src/poly/fattn_mma.cuh` is llama.cpp's `fattn-mma-f16.cuh` (MIT) from the pinned llama.cpp commit,
with three marked changes: the causal mask is computed from the positions instead of read from memory, and a tile
writes its softmax max and sum for each token and head. `src/poly/fattn.cu` launches the D=256 kernel with 8 tokens
of a group's 8 heads per tile (GQA 8:1, Ampere or later), one CUDA block per tile. The keys go in blocks. Each
block's K and V rows are dequantized from the q8_0 cache to fp16, the kernel attends to them, and each block's
results are merged into the ones before it by log-sum-exp. The last block holds the batch's own rows and starts at
or before the batch's first position. Everything lives in the attention's existing score scratch (64 MiB: a
block's results plus 24K rows of K and V for a 1,024-token batch), so no VRAM is added. The server loaded with
1,987 experts in VRAM. It is opt-in: `STRATA_FATTN=1` in the environment, and the server logs
`STRATA_FATTN: prompt attention through llama.cpp's flash-attention kernel`. Without it nothing changes. Decode
(fewer than `kStageRows` tokens) keeps its own attention kernels.

**Numerics.** `poly_fattn_test` (`src/poly/fattn_test.cu`) runs on the 3070 with random q8_0 caches of
Qwen3.6's shape. It compares the new path with `attend_batch()` and both with the arithmetic in doubles (three tokens
a case). The cases run from 16 tokens at position 0 to 1,024 at 120,000, plus two with a small scratch that forces 8
and ~190 key blocks. All pass. The largest difference from `attend_batch()` is 1.9e-2 on results up to 2.9 (mean
1.3-2.9e-4). Against the doubles the new path is within 7.7e-3, `attend_batch()` within 2.3e-3. The new path is
further off because llama.cpp's kernel adds the value products up in halves (`attend_batch()` uses floats). Smaller
key blocks bring it closer (6.0e-3 with 4K-key blocks against 1.9e-2 with 24K). On the model: the 107K recall
test passed (3 of 3 facts). Greedy answers (thinking off, 300 tokens) to a ~27K and a ~107K prompt part from the
old path's after 14 and 72 characters. Both answers are coherent and on topic. The old path run twice parts from
itself at the same two places, so these are near-ties that flip between runs anyway.

**Speed.** The kernel alone, one layer, a 1,024-token batch (the test, with the idle server sharing the card): 0.27
ms against 1.91 at position 0, 11.1 against 31.4 at 30,000, 42.7 against 122.3 at 120,000 (about 47 TFLOPS against
16; above the ~40 of FP32 sums because the value products sum in halves, which the tensor cores do twice as fast). On the model, the Pass 8 sequence (restart, the 107K recall warm-up, `multiturn.py fwd 5`), RAM 3200, Plex and
the add-on generator running. The control is Pass 9's control run (the measuring build without changes, earlier the
same day). At the user's choice no second control was run:

| Turn, context | Read tok/s, Pass 9 control (range of Pass 9 runs) | Read tok/s, `STRATA_FATTN=1` | Prompt s, control / flash | Decode tok/s, control / flash |
|---|---|---|---|---|
| 1, 32K | 1,437 (1,437-1,586) | 1,824 (+15-27%) | 22.0 / 17.4 | 52.3 / 52.8 |
| 2, 72K | 1,022 (988-1,094) | 1,465 (+34-48%) | 39.8 / 27.9 | 41.5 / 41.2 |
| 3, 103K | 749 (728-769) | 1,268 (+65-74%) | 40.6 / 24.2 | 35.1 / 34.6 |
| 4, 143K | 610 (577-625) | 1,066 (+71-85%) | 65.2 / 37.3 | 30.6 / 29.5 |
| 5, 174K | 496 (466-511) | 1,052 (see below) | 63.4 / 64.6 | 26.7 / 26.2 |

Turn 5 read 67,929 tokens this time, against about 31K in every earlier run. About 36K tokens were read again, as in
Pass 7's run A turn 6 (the prompt cache's checkpoint was far back). So it read twice the tokens in the same time,
starting from 106K. Its rate is over a lower average context than the control's, and its time isn't comparable.
The answers differ from the control's from turn 1 on (greedy near-ties, above), so the prompt sizes differ by a few
hundred tokens. Decode is the same within 1 tok/s. The recall warm-up passed.

So the 30-40K-token follow-ups at 100-145K read in 24-37 s instead of 41-65 s. The gain grows with the context,
as the profile says it should.

Measuring build: image `strata-cybertiel:p10` on Tower (`/mnt/user/appdata/polystrata/pass10/`: the changed
sources, `p10run.sh`; results `/data/pass10-fattn.txt`, `/data/pass10-cmp.py`, `/data/pass10-*-answers.json`;
profile `/data/profile/p9-*`). `strata-cybertiel:p9` is removed. Tower runs its normal image and config, without
`STRATA_FATTN`.

Not tested: the fp16 KV cache (the new path takes q8_0 only), other models, Turing (the kernel has no 64-column tile
there, and the old path runs), a HIP build (`fattn.cu` is compiled out there and the old path runs; not built here),
and a same-day control run.

### Next

1. Done (2026-10-18): Tower runs it. `strata-cybertiel:latest` is now the old `latest` plus the `fcb7c9aa`
   sources (`CMakeLists.txt`, `src/poly/`) and an engine rebuild (`/mnt/user/appdata/polystrata/fattn/Dockerfile`).
   The image's other sources match the branch. The old image is tagged `strata-cybertiel:pre-fattn` as a rollback,
   and `:p10` is kept. The container has no dockerMan template (it was made with `docker create`), so it was
   recreated with the Pass 7 command plus `-e STRATA_FATTN=1`:

   ```bash
   docker create --name strata-cybertiel --gpus all -p 8080:8080 --ulimit memlock=-1 --stop-timeout 60 --restart no \
     -v /mnt/user/appdata/polystrata/data:/data -v /mnt/cache/llm:/gguf \
     -e FAMILY=cybertiel -e MODEL=UD-Q4_K_XL -e GGUF_DIR=/gguf -e CONTEXT=262144 -e KV=int8 -e VISION=cpu -e GPU=0 \
     -e STRATA_FATTN=1 strata-cybertiel:latest
   ```

   The log shows the `STRATA_FATTN` line. The engine loaded in 9.3 s with 1,988 experts in VRAM and the container
   reports healthy. The 107K recall test passed (3 of 3 facts, 107,399 tokens read in 72 s from cold).
2. If the halves' error matters for some use, cap the key blocks (4K keys gave 6.0e-3 against 1.9e-2) and measure
   the cost of the extra merges.

## Pass 11: VRAM audit for the expert cache (2026-10-18)

Goal: more routed experts in the 3070's VRAM, so that fewer are read by the CPU (Pass 6: 57-66% of a decode token).
Live config: 256K + `--kv q8_0` + `STRATA_FATTN=1` (Pass 10), 12 threads, RAM 3200. Each measurement is the Pass 8
sequence (restart, the 107K recall warm-up, `multiturn.py fwd 5`), with Plex, Deluge and the add-on generator
running. Results: `/data/vram-ab-{1,2,3,4}.txt`. Scripts: `/mnt/user/appdata/polystrata/vab.sh` (A/B, it restores
`.q8-256k` on exit), `vsweep.sh` (VRAM per setting), `recreate.sh` (the Pass 10 `docker create`, image as argument).

### Where the VRAM goes

At idle the engine holds 7,464-7,526 MiB of the 8,192 (`nvidia-smi`), and 511 MiB is free. From the GGUF and from
what each setting gives the expert cache when it changes:

| Part | MiB | How it was measured |
|---|---|---|
| Expert cache (1,988 experts) | 3,631 | `/metrics` `expert_cache_mib` |
| Weights the card holds | ~2,080 | GGUF tensor sizes: attention and gate projections 1,041, `output.weight` 515, linear-attention (ssm) 274, shared experts 128, routers 80, the MTP block's dense part 41 |
| Expert staging buffer (`--stage-mib`, one layer's experts here) | ~560 | `--stage-mib 256` gave the cache 306 MiB more |
| Prompt-batch buffers (`--prompt-batch 1024`) | ~440 | 512 gave 222 MiB more, 256 gave 333 |
| Kept free (the reserve) | 511 | `vram_free_mib`. The engine finds RAM "tight" (available RAM within 4 GiB of the model's files: 25.7 GB against 22.7 + 4), which makes the reserve 512 MiB and keeps all of it free (`default_reserve_mib()`, `mind_card_vram()`). |
| The rest: CUDA context, cuBLAS work space, recorded graphs, attention scratch (64 MiB), short KV, the token table if it is on the card | ~750 | the difference |

Flash attention (Pass 10) reuses the existing attention scratch, so it changed nothing here.

### Freeing VRAM: what it costs and gives

Expert-cache hit rate and decode tok/s for turns 1-5 (32K to 174K), mean of two runs each, first round
(`vram-ab-2.txt`):

| Setting | Experts in VRAM | Hits | Decode tok/s | Read tok/s |
|---|---|---|---|---|
| base | 1,988 | 50 / 41 / 41 / 36 / 31% | 46.2 / 35.8 / 31.2 / 26.9 / 24.8 | 1,740 / 1,527 / 1,222 / 1,108 / 972 |
| `--stage-mib 256` | 2,155 | 50 / 41 / 44 / 34 / 34% | 44.8 / 36.4 / 31.8 / 26.7 / 24.7 | 509 / 488 / 464 / 438 / 404 |
| `--prompt-batch 512` | 2,249 | 56 / 48 / 48 / 40 / 38% | 48.8 / 37.5 / 33.5 / 28.5 / 25.2 | 1,467 / 1,238 / 1,038 / 946 / 836 |

- **`--vram-reserve-mib 256`** (256 MiB kept free, +139 experts): the 107K recall test failed with
  `cudaGraphInstantiate: out of memory`, so the reserve is what recording the launches needs at long context.
  Out. Combined with the two settings below, prompt reading fell to 250-295 tok/s (`vram-ab-1.txt`).
- **`--stage-mib 256`** (+167 experts): no decode gain, and prompt reading 3x slower. A layer's uncached experts
  no longer fit the buffer in one round, so each batch copies them over PCIe 3.0 in several. Out.
- **`--prompt-batch 512`** (+261 experts; it also kept 255 MiB free, not 511, presumably because RAM was not "tight"
  at that load): hits +5-7 points, decode +1-7%, prompt reading 14-19% slower. Per turn (wall time) it comes out
  even. Not kept.
- So more experts help decode, but by about 1 tok/s per 100 experts at mid context. And the VRAM these settings free
  is what prompt reading uses.

### Expert swaps

`--expert-swaps N` moves up to N experts at a time into the cache while an answer is written, in place of cached
ones the answer uses less. Hits for turns 1-5 (`vram-ab-3.txt`, `vram-ab-4.txt`):

| Swaps | Hits |
|---|---|
| 0 | 17 / 16 / 16 / 14 / 10% |
| 32 | 49-53 / 37-42 / 40-41 / 36 / 31-33% |
| 64 (default) | 50-53 / 38-42 / 41-43 / 35-36 / 28-33% |
| 128 (a build with the cap at 256) | 46-49 / 40-47 / 41-44 / 35-36 / 31% |
| 256 (same build) | 50-53 / 36-43 / 39-42 / 34-36 / 28-32% |

1. **Swaps make the hit rate.** The experts the profile picks at load cover 10-17% of this coding traffic's uses.
   Following the conversation brings it to 30-53%. Decode without swaps was 20-32 tok/s against 25-46 with them.
2. **The cap of 64 is not the limit.** 32 to 256 give the same hit rates. The engine already swaps in what the
   answer uses more than what it holds (`follow()`: twice as much recent use, plus a minimum). The 256 build was a
   one-line change to the check in `Engine::load()` (0 to 64), rebuilt as image `strata-cybertiel:swaps`. It was not
   kept, and the image is removed.
3. Decode in `vram-ab-4.txt` is not comparable between rows: during those runs an `rsync` read through `/mnt/user`
   (`shfs` at 153 MB/s), `kswapd0` and `kcompactd` were busy, and strata-poly read 190 MB/s from disk to bring its
   evicted pages back. The base runs there decoded at 20-35 tok/s and read prompts at 1,244-1,674 tok/s.
4. The expert profile is written back after each request (the file's time on the host is Pacific, in the container
   UTC). It holds lifetime counts, most of them from the benchmark's repeated manual text, but with swaps doing the
   work this matters little: the load-time pick is worth 10-17% of the hits.

### Verdict

Nothing kept; Tower runs as before (`strata-cybertiel:latest`, `.q8-256k`, `STRATA_FATTN=1`). The levers left for
more experts at no prompt-reading cost need code:

- Prompt-batch buffers that the expert cache borrows outside a prompt read (about 440 MiB, 240 experts; Pass 11's
  `--prompt-batch 512` row is what about half of it bought: +1-7% decode). Like PR #46 (VecSzn/PolyStrata) does for
  the staging buffer. The staging buffer itself (#46's case) gave no decode gain here.
- A reserve that is only kept while launches are being recorded, if graph recording can be made to happen once.

Both are worth at most a few percent of decode on this card. The larger effect seen in this pass is outside the
engine: other disk traffic on Tower evicts the model from the page cache.

## Pass 12: a decode attention kernel (2026-10-18)

At long context a coding turn's answer was the slow part after Pass 10: decode fell from about 52 tok/s at 32K to 27
at 174K.

**Profile.** Nsight Systems as in Pass 10, but with CUDA graphs traced per kernel (`--cuda-graph-trace=node`:
decode runs in recorded graphs, and Pass 10's capture counted only the launches outside them). A ~138K and a ~170K
conversation (`long_prompt`), then the same prompt again, all from the prompt cache, with 300 tokens of answer,
thinking on; only that answer captured. The server on its normal config (`STRATA_FATTN=1`), `strata-cybertiel`
stopped meanwhile. Scripts `/mnt/user/appdata/polystrata/pass12/{nsys12.sh,cap12.py}`, captures
`/data/profile/p12-{140,170}.*`.

| | ~138K | ~170K |
|---|---|---|
| answer | 161 tokens at 25.1 tok/s, 1.8 tokens a pass (70 of 86 guesses right) | 156 tokens at 26.8 tok/s |
| capture, GPU busy | 6,463 ms, 3,498 ms | 5,864 ms, 3,645 ms |
| `attn_value_t_kernel` | 954 ms (862 us a launch) | 1,038 ms (1,067 us) |
| `attn_softmax_kernel` | 686 ms (588 us) | 770 ms (753 us) |
| `attn_score_reg_kernel` | 547 ms (469 us) | 621 ms (607 us) |
| `attn_value_kernel` (4-token windows) | 205 ms (3,411 us) | 212 ms (4,230 us) |
| **decode attention, share of the capture** | **2,392 ms, 37%** | **2,641 ms, 45%** |
| the expert products (`native_mmvq_*`) | 811 ms | 742 ms |

About 20 ms of a ~55 ms pass at 138K. A pass's 10 attention layers read 10 x 138K rows x 1,088 bytes (K and V,
q8_0) = 1.5 GB, 3.4 ms at the 3070's 448 GB/s, so the old kernels ran about 6x off the bandwidth bound. They write
every score as a float, read them back for the softmax and again for the values, and the value kernel runs 16 to 32
CUDA blocks on the 46 multiprocessors.

**The change.** `fdec_q8()` in `src/poly/fattn.cu`, a kernel of this project's own (llama.cpp's `fattn-vec` packs
no GQA heads, so it would read the cache once per head). One CUDA block takes 1,024 keys of a KV head's group for all
of the call's tokens (up to 4: 32 query rows, the group's 8 heads per token), as flash attention: per 64 keys the
scores on the tensor cores (`mma.sync` m16n8k16; queries as halves, key codes exact as halves, each 32-value block's
scale applied to its sum in floats), the softmax max and sum carried along per row, and the values' products on the
tensor cores (weights and dequantized values as halves, sums in floats). The keys are read once per 4 tokens. A second
kernel merges the 1,024-key splits. The splits are counted from the cache's first row and no row's arithmetic
depends on the others, so a token gets the same bits alone or in a window, as with the old kernels. The splits'
results live in the attention's existing score scratch: no VRAM is added (both variants below held 1,987-1,988
experts with 511 MiB free). Opt-in: `STRATA_FDEC=1` in the environment (q8_0 cache, heads of 256, 8 heads a group,
Ampere or later); the log says `STRATA_FDEC: decode attention through fdec_q8()`. Without it nothing changes.

**Numerics.** `poly_fattn_test dec` on the 3070 (random q8_0 caches of this shape, 172K rows): fdec_q8() against
attend() and both against doubles, for 1 to 6 tokens at positions 0 to 170,000, including the split edges (1,022 to
2,047). All pass: at most 1.8e-3 from the doubles on results up to 4 (attend(): about 1e-6), from the halves. Each
token alone, in a window, and with its position read from the card (the recorded path) gives the same bits. On the
model: the 107K recall test passed in all runs below. Not run: the 247K needle (see Next).

**Speed.** The kernel alone, one layer (attend() in brackets): 0.45 ms (1.92) for 1 token at 140,000, 0.51 (4.91)
for 3 tokens, 0.67 (6.33) for 4 at 170,000. That is about 340 GB/s, 76% of the bandwidth. Splits of 512 and 256 keys
were 7-60% slower at long context. Below about 4K of context the new kernel is the slower one (0.06 against 0.03 ms
a layer at 1-2K: too few splits to fill the card), which the short requests below don't show.

On the model, the Pass 8 sequence with 3 short requests first (~900 tokens of prompt, 400 of answer, thinking off),
Plex, Deluge and the add-on generator running, two runs of each, alternating. Both variants ran `--vram-reserve-mib
512` (below). Results `/data/pass12-ab.txt` (`pass12-ab-1.txt`: the first attempt). Decode tok/s:

| Turn, context | Base (2 runs) | `STRATA_FDEC=1` (2 runs) | Change of the means |
|---|---|---|---|
| short (3 requests each) | 66.1-77.7 | 61.8-77.2 | none (first request of a run 62-68 in both) |
| 1, 32K | 52.0 / 55.6 | 64.0 / 63.7 | +19% |
| 2, 72K | 40.6 / 36.0 | 55.1 / 55.3 | +44% |
| 3, 103K | 34.4 / 35.7 | 52.6 / 53.4 | +51% |
| 4, 143K | 29.3 / 29.3 | 49.7 / 47.7 | +66% |
| 5, 174K | 27.0 / 26.2 | 45.1 / 46.0 | +71% |

Prompt reading is unchanged (it doesn't use this path). A ~1,000-token answer at 140-175K now takes about 20 s
instead of about 35-40. Turn 5 read ~68K tokens again in one run of each (the prompt-cache checkpoint of Pass 10).

**The reserve flips.** The first A/B attempt failed: the fdec run loaded with 2,128 experts, not 1,988, and its
first request ended in `cudaGraphInstantiate: out of memory` (177 MiB free). That was not the kernel. The engine
keeps 512 MiB free when RAM is "tight" for the model and 256 otherwise (Pass 11), and Tower sits on that line: this
load came out not tight (24 GiB available). Pass 11 saw 256 MiB fail graph recording with the old kernels too. The
swap file (above) raises `MemAvailable`, which makes the flip likelier. So the A/B pinned `--vram-reserve-mib 512`
(with a reserve named, the engine keeps all of it, as when RAM is tight).

Measuring build: image `strata-cybertiel:fdec` (`strata-cybertiel:latest` plus the changed `src/poly/`, engine
rebuilt; `/mnt/user/appdata/polystrata/pass12/`: `Dockerfile`, `p12ab.sh`, `p12test.sh`, `repo/`).

**On the upstream branch.** The kernel was then moved onto `strata-poly/fattn-prompt` (the prompt kernel prepared for
upstream, with the fp16 cache and the per-card checks), as branch `strata-poly/fdec-decode` (commit `6b977223`, not
pushed). There Qwen35 decides once whether it runs (`fdec_fits()`: the card, and the scratch for the whole context)
and says so, as that branch does for `STRATA_FATTN`. `poly_fattn_test dec` gives the same errors and times, and the
branch's prompt cases pass for q8_0 and fp16. Image `strata-cybertiel:fdec-b` (`latest` plus that branch's
`src/poly/` and `CMakeLists.txt`: the engine is that branch's, setup and server stay this branch's). One pair,
`/data/pass12-ab-b.txt`, base being that branch's build without `STRATA_FDEC`:

| Turn, context | Base | `STRATA_FDEC=1` |
|---|---|---|
| short (3 requests, ~1,160 tokens of prompt) | 61.6 / 75.5 / 77.7 | 64.3 / 72.6 / 69.1 |
| 1, 32K | 52.4 | 49.9 |
| 2, 72K | 40.5 | 52.8 |
| 3, 103K | 35.9 | 52.8 |
| 4, 143K | 30.2 | 48.5 |
| 5, 174K | 25.9 | 45.3 |

Turns 2-5 match the runs above within 2.5 tok/s. Turn 1 came out at 49.9, where both runs above gave 64: the hit
rate was the same as base's (51%), and turn 1 is the one that varies most between runs (Pass 8), but this is one run
and it isn't explained. The 107K recall test passed in both; 1,987 experts and 511 MiB free in both.

Not tested: Turing (the `mma` shape needs Ampere: the old kernels run), the fp16 KV cache, other head shapes, a HIP
build (compiled out there), the multi-sequence path (`attn_rows()` takes the kernel too, but no run used it), more
than one run per variant per day.

**Quality.** The server gives no token probabilities, so: 15 questions after a prefix of this repository's C++/CUDA
source (39K, 115K and 196K tokens; at each size 4 ask a constant's value at 5-90% depth, checked against the code,
and 1 asks what the last function does), temperature 0, thinking off, 250 tokens. Image `:fdec-b`, the live 256K
q8_0 config; "old" is both flags off, "new" both on (`q12.sh`, `qual12.py`, results `/data/profile/qual-*.json`,
log `/data/pass12-qual.txt`). The fourth run (a second "new") failed to start in time (see Next) and was not redone.

| | Right (of 12) | Identical answers (of 15) | Characters alike before the first difference (median) | Similarity (difflib, mean) |
|---|---|---|---|---|
| old 1, old 2 | 11, 11 | | | |
| new | 11 | | | |
| old 1 against old 2 | | 1 | 58 | 0.60 |
| old against new | | 0 | 67-71 | 0.56-0.59 |

All three runs miss the same question (`640` for a 64 at 115K) and get the other 11 right. The new kernels' answers
part from the old ones' no earlier than two runs of the old kernels part from each other (the old path is not
deterministic from run to run: the CPU experts' sums). That is what the numerics predict, not proof that no answer
anywhere gets worse.

**fp16 cache.** Branch commit `d7ff87b3` adds `fdec_f16()`: the same kernel reading an fp16 cache where it lies
(`poly_fattn_test dec`: all cases pass for q8_0 and fp16; one layer at 140,000 with fp16: 0.72 ms against 1.82 for
one token, 0.78 against 4.55 for three, about 89% of the bandwidth). On the model: image `:fdec-b2`, the config
without `--kv q8_0` and with `--max-context 131072` (fp16 at 256K leaves the expert cache too little), one run each
(`f16ab.sh`, `/data/pass12-f16.txt`). Read / decode tok/s:

| Turn, context | Old | `STRATA_FATTN=1` | Both |
|---|---|---|---|
| short (3 requests) | - / 68.4-70.7 | - / 72.0-72.2 | - / 67.3-77.2 |
| 1, 32K | 1,467 / 53.0 | 1,728 / 53.2 | 1,735 / 62.6 |
| 2, 72K | 997 / 38.3 | 1,491 / 40.5 | 1,487 / 49.6 |
| 3, 103K | 764 / 32.5 | 1,324 / 33.4 | 1,227 / 43.2 |

So with fp16 the prompt kernel reads 18-73% faster and the decode kernel decodes 18-33% faster than the old path
(less than with q8_0: twice the bytes, and the old fp16 path was nearer its bound). The 107K recall test passed in
all three; 1,990 experts and 511 MiB free in all three.

Again at 256K with 5 turns (`f16ab2.sh`, `CTX=262144 TURNS=5`, `/data/pass12-f16-256k.txt`), to reach 143K and
174K. Read / decode tok/s:

| Turn, context | Old | `STRATA_FATTN=1` | Both |
|---|---|---|---|
| short (3 requests) | - / 64.4-76.9 | - / 66.1-78.2 | - / 66.3-75.9 |
| 1, 32K | 1,465 / 55.7 | 1,844 / 56.3 | 1,856 / 62.0 |
| 2, 72K | 1,041 / 39.2 | 1,470 / 35.7 | 1,492 / 50.2 |
| 3, 103K | 772 / 33.3 | 1,287 / 34.0 | 1,209 / 45.9 |
| 4, 143K | 597 / 25.6 | 1,109 / 26.0 | 1,116 / 36.9 |
| 5, 174K | error | error | error |

Turn 5 ended in HTTP 400 in all three, the old path included. The log went with the container, so the cause is not
confirmed; an fp16 cache at 174K takes ~3.6 GB, and at 143K the expert cache had 3-7% hits (nearly nothing left
of it). Up to 143K: reading +26-86%, decode with both against the prompt kernel alone +10-42%.

**Qwen3.6-35B-A3B.** The stock model the code is written for (`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, Unsloth's file
without the MTP layer, so every decode step is one token), same image `:fdec-b2`, 256K + q8_0 + reserve 512 +
its family's `--tail-skip 6`, in a container `strata-q36` with `strata-cybertiel` stopped; its config and expert
profile were put back afterwards (`q36.sh`, `/data/pass12-q36.txt`). One run each. Decode / read tok/s:

| Turn, context | Both flags off | Both on |
|---|---|---|
| short (3 requests) | 74.9-82.5 / - | 73.8-81.7 / - |
| 1, 32K | 55.0 / 1,433 | 65.8 / 1,839 |
| 2, 72K | 41.2 / 1,035 | 59.7 / 1,502 |
| 3, 103K | 34.5 / 822 | 53.8 / 1,387 |
| 4, 143K | 29.1 / 659 | 49.1 / 1,203 |
| 5, 174K | 25.7 / 537 | 45.1 / 1,069 |

Decode +20% at 32K to +75% at 174K, reading +28% to +99%; the 107K recall test passed in both, 2,143 experts and
511 MiB free in both. The two models do not both fit in the page cache: the first run started with 17.9 of the
file's 20.6 GiB cached, the second with 20.2 (hit rates the same, 47-72% in both).

### Next

1. Done (2026-10-18): Tower runs it, from the upstream branch's build. `strata-cybertiel:latest` is now `:fdec-b`;
   the old `latest` (the `fcb7c9aa` prompt kernel) is tagged `:pre-fdec` as the rollback. The config and the
   `.q8-256k` reference have `"--vram-reserve-mib", "512"` (the old reference is `.q8-256k.pre-r512`).
   `recreate.sh` adds `-e STRATA_FDEC=1` (old copy `recreate.sh.pre-fdec`), and the container was recreated with
   it: the log shows both `STRATA_FATTN` and `STRATA_FDEC` lines, 1,987 experts in VRAM, healthy. The 247K needle
   (`TOKENS=185000`, seed 5) passed, 3 of 3 facts from 247,375 tokens in 213 s (Pass 4: 409 s); GPU memory at most
   7,627 MiB, afterwards 1,987 experts and 511 MiB free, no restart, no error in the log. To roll back:
   `recreate.sh.pre-fdec strata-cybertiel:pre-fdec` and `.q8-256k.pre-r512` as the config.
2. Turn 1 of the `:fdec-b` pair: one more run would tell whether it was noise.
3. (Fixed in item 4.) Every new container from an image built this way compiled the Qwen3.8 engine again at its start (~4 minutes):
   the Dockerfiles rebuild only strata-poly, and setup then finds the engine's source hash changed. It does not
   touch the measurements (it ends before the server starts), but a recreate takes ~5 minutes; `recreate.sh` now
   waits up to 10. Building both engines in the Dockerfile would fix it.
4. Done (2026-10-19): Tower runs exactly the code of the upstream PRs (VecSzn/PolyStrata #50 and #51,
   `strata-poly/fdec-decode` at `67f60c12`). Image `strata-cybertiel:pr51` (`pass12/Dockerfile.pr51`: `:pre-fdec`
   plus that branch's `CMakeLists.txt`, `cmake/`, `include/` and `src/`; it builds strata-poly and the Strata engine
   and writes the Strata engine's new source hash, so a new container no longer compiles at its start) is now
   `:latest`; the previous live build stays as `:fdec-b`, and `:pre-fdec` is the rollback without the kernels.
   Recreated with `recreate.sh`: healthy in 15 s (was ~5 minutes), both kernel lines in the log, 1,988 experts,
   511 MiB free. The 107K recall test passed; decode 62.2 / 54.8 / 51.7 tok/s at 32K / 72K / 103K
   (`multiturn.py fwd 3`), as the runs above.

## Pass 13: faster CPU expert reads (2026-10-18/19)

[PLAN-cpu-experts.md](PLAN-cpu-experts.md) proposed three ways to speed up the CPU expert reads (57-66% of a decode
token, Passes 6 and 11): (A) huge pages, (B) software prefetch, (C) repacked Q4_K rows. Upstream PR 36 (Xiaomi2008,
open) turned out to be B: `_mm_prefetch` two rows ahead and down tasks of 32 rows instead of 16, in
`src/poly/cpu_experts.cpp`. Its exact change was tested rather than one of our own. **Outcome: none of the three is
kept.** Tower runs a rebuild with upstream PR 34 and PR 46 and no CPU-expert change. Sources and scripts:
`docs/tower/p13/` (on Tower `/mnt/user/appdata/polystrata/p13/`), results `/data/pass13-*.txt`.

### Phase 0: rebuild with PR 34 and PR 46

The local `tower` branch (`64f17cc2`: origin/main, which has PR 34 `--skip-weight` and PR 46 `--stage-read-mib`, plus
PR 50/51 and `prefill`) as image `strata-cybertiel:tower-64f17cc2` (`p13/Dockerfile.tower`: the `Dockerfile.pr51`
recipe on `tower`'s sources, both engines built). `polystrata:tower` is rebuilt `FROM` it (build dir
`/mnt/user/appdata/polystrata/tower`, old Dockerfile `Dockerfile.pr51`). The container was recreated from its template
with `/usr/local/emhttp/plugins/dynamix.docker.manager/scripts/rebuild_container PolyStrata`, then `docker start`
(it is not autostarted). It was healthy in 6 s, with no compile at start, both kernel lines (`STRATA_FATTN`,
`STRATA_FDEC`), 1,988 experts and 511 MiB free.

The Pass 12 sequence (restart, 3 short requests, the 107K recall warm-up, `multiturn.py fwd 3`). The control is the
`:pr51` image, run the same evening just before. Decode tok/s (read tok/s):

| Turn, context | `:pr51` (control) | `tower-64f17cc2` | `tower-64f17cc2` in Phase 3 (2 runs) | + `--skip-weight 0.06` |
|---|---|---|---|---|
| short (3 requests) | 65.2 / 72.2 / 70.6 | 66.4 / 68.6 / 71.7 | 62.0-75.1 | 69.0 / 69.1 / 71.9 |
| 1, 32K | 65.1 (1,703) | 63.6 (1,717) | 66.1 / 66.5 | 67.2 (1,726) |
| 2, 72K | 55.8 (1,557) | 54.2 (1,551) | 59.5 / 59.3 | 57.2 (1,480) |
| 3, 103K | 53.4 (1,280) | 49.9 (1,275) | 53.6 / 52.3 | 55.0 (1,283) |

1. **The rebuild matches `:pr51`.** The first run's turn 3 was 7% lower. The two Phase 3 runs of the same image
   came out at or above the control, so that was run-to-run noise. Both new options leave the default path as it
   was (with the option at 0, `widen_stage()` and the weight check return at once). The recall test passed in every
   run.
2. **`--stage-read-mib` can't do anything here.** It grows the staging buffer for a prompt read, up to one layer's
   experts, and Tower's buffer already holds a layer (Pass 11: ~560 MiB; a layer's 256 experts are 476 MiB). Not run.
3. **`--skip-weight 0.06`** (the lowest-KL setting in [BACKENDS.md](../BACKENDS.md), measured there on GLM 5.3 Flash) was run
   once: +3-10% decode against the same evening's base runs, about the run-to-run spread. It changes the answers
   (they were shorter here: 900-1,430 characters against 1,350-1,700). It would need a quality check on this model
   and more runs before it could be called a gain. Not turned on.

### Phase 1: microbenchmark

`p13/cxb.cpp` does a decode token's CPU expert work through the engine's own `cpu_experts()`, linked in, on the
real GGUF's bytes. That is 40 MoE layers with 8 experts each, every expert on the CPU (which the engine never
does: about half are in VRAM), 12 threads, drawn at random from 128 experts per layer (9.4 GiB). It is linked
against ggml-cpu from the image's pinned llama.cpp, built native as the engine's is (`p13/cxb.sh`). `ROWS` is the
number of tokens in a call. A decode window of the known token plus MTP and lookup guesses gives 1-4. PolyStrata was
stopped, while Plex, Deluge and the add-on generator ran. Each variant ran 3 times, interleaved, with 20 warm-up
tokens and 200 timed tokens.

Types in this file: gate and up are Q4_K in 39 of the 40 layers (one is Q5_K), and down is Q5_K in 36 and Q6_K in 4.
The pinned ggml has an AVX2 `ggml_gemv_*_8x8_q8_K` only for Q4_K (Q5_K and Q6_K have the generic C), so C covers
gate and up, 62% of the expert bytes. THP on Unraid is `always`, defrag `madvise`.

ms per token, mean of 3 (range):

| Variant | 1 token a call | 2 | 3 | 4 |
|---|---|---|---|---|
| today: `vec_dot`, rows from the file's mapping | 19.67 (19.66-19.67) | 42.06 (41.84-42.21) | | |
| same, second session | 20.47 (19.83-21.48) | 44.07 (42.74-44.82) | 68.45 (66.13-69.82) | 87.15 (85.72-89.96) |
| anonymous copy, 4 KiB pages | 19.47 | 41.87 | | |
| **A**: anonymous copy, `MADV_HUGEPAGE` (53-75% of it in 2 MiB pages) | 19.50 | 41.89 | | |
| **A** at 100% 2 MiB pages (page cache dropped, memory compacted first) / its 4 KiB control | 19.00 / 19.20 | 41.85 / 40.58 | | |
| **B**: PR 36, from the mapping | 21.31 | 38.72 | | |
| **B**, second session | 21.89 | 40.58 | 59.27 | 75.77 |
| PR 36's prefetch alone (down tasks left at 16) | 22.19 | 41.46 | 63.47 | 76.81 |
| **B** on A | 21.01 | 37.82 | | |
| **C**: Q4_K gate and up as `block_q4_Kx8`, `ggml_gemv_q4_K_8x8_q8_K`, on A | 19.80 | 43.08 | | |

Today's 19.7 ms for 615 MB is 31 GB/s, against 44-46 GB/s for a sequential read (Pass 8).

1. **A: no gain.** At full 2 MiB coverage it is 1% faster at 1 token a call and 3% slower at 2. So the TLB is not
   what holds the reads back. On a host up for 23 hours, `MADV_HUGEPAGE` got only 53-75% of the copy into huge pages,
   even after compaction. An engine using A would see that coverage too.
2. **C: no gain** (+1% at 1 token, +2% at 2). The plan's reason for C was wrong. A task already reads its 32 rows
   of gate (or up) as one contiguous 36 KiB block. Interleaving 8 rows reorders the bytes inside that block, so
   nothing gets more contiguous. Loading the activation once instead of 8 times doesn't matter when the reads set
   the pace. The repacked results were within 4e-8 of `cpu_experts()` on the file's rows (largest output 0.2).
3. **B: depends on the tokens per call.** Compared within each round: +6 to +9% (slower) at 1 token, -7 to -10% at
   2, -12 to -14% at 3, -13% at 4. The prefetch alone does about the same, so the slowdown at 1 token comes from the
   prefetches, not from the larger down tasks. With one use per expert row, the prefetch instructions cost more than
   they save. Output is bit-identical to today's: the same FNV hash over every call's output, at 1, 3 and 4 tokens a
   call (`pass13-bench2.txt`, "parity").
4. Against the stop rule (10% of the CPU expert time): only B passes, and only at 3-4 tokens a call.

### Phase 2 and 3: PR 36 in the engine

PR 36's commit was cherry-picked onto `tower` (`6ec61d5c`) and built as `strata-cybertiel:tower-6ec61d5c` /
`polystrata:tower-b36`. There is no flag, since it changes no output. A/B (`p13/p13ab.sh`, `pass13-ab.txt`): a
container `strata-p13` with the template's settings, PolyStrata stopped meanwhile, and runs alternating base and B.
Each run was 3 short requests, the 107K recall warm-up, then `multiturn.py fwd 5` or `rev 5`, two runs of each. Plex,
Deluge and the add-on generator ran throughout. Decode tok/s, mean of 2 (the two runs):

| Turn | fwd context | base | PR 36 | rev context | base | PR 36 |
|---|---|---|---|---|---|---|
| short (6 requests) | | 69.8 | 68.1 | | 69.1 | 69.1 |
| 1 | 32K | 66.3 (66.1 / 66.5) | 62.9 (63.6 / 62.2) | 35K | 61.5 (55.8 / 67.3) | 63.4 (67.5 / 59.3) |
| 2 | 72K | 59.4 (59.5 / 59.3) | 53.2 (54.4 / 52.0) | 77K | 56.2 (57.1 / 55.2) | 56.6 (55.0 / 58.2) |
| 3 | 103K | 53.0 (53.6 / 52.3) | 49.9 (51.4 / 48.4) | 123K | 51.2 (50.7 / 51.7) | 51.3 (50.5 / 52.1) |
| 4 | 142K | 49.0 (49.3 / 48.8) | 46.8 (46.5 / 47.1) | 204K | 42.1 (42.1 / 42.1) | 40.9 (40.5 / 41.2) |
| 5 | 174K | 47.7 (47.1 / 48.2) | 44.5 (45.5 / 43.5) | 240K | 38.0 (38.2 / 37.8) | 39.1 (37.1 / 41.1) |

1. **PR 36 gives no decode gain in the engine.** On fwd it was slower at every turn in both runs: -4 to -10%, and
   -10% at 72K, where the two runs of each agree within 2.4 tok/s. On rev it was within 3%. The keep rule (+5%
   across the turns) fails, so it is not kept, and `tower` reverts it (`4d741b5b`, content = `64f17cc2`). As in
   Pass 8, the microbenchmark's gain didn't carry over. The engine's calls are a mix of 1-token and longer windows,
   about half of a token's experts are in VRAM, and the GPU's work overlaps the CPU's. Which of these cancels it was
   not measured.
2. Everything else was the same for both: loaded in 2.5 s, 1,988 experts and 511 MiB free after each run, the
   engine's RSS 24.6-24.7 GB (mostly the mapped file's cached pages), the file 21.2 of 21.2 GiB cached after every
   run, and every recall test passed. Start-up time and RAM don't change with B, so the cold/warm start-up test the
   plan asked for (meant for A and C's anonymous copy) was not run.
3. The 247K needle (`TOKENS=185000`, seed 5) on the live image after the A/B: 3 of 3 facts in 213 s (Pass 12: 213
   s). GPU memory at most 7,513 MiB. Afterwards 1,988 experts, 511 MiB free, no restart, no error in the log.
4. Not run: the Pass 12 quality check. B's output is bit-identical in the microbenchmark, and B isn't kept. The live
   image has PR 34/46 off, so its arithmetic is `:pr51`'s.

### What ran after Pass 13 (superseded by Pass 14)

- `PolyStrata` (template unchanged) on `polystrata:tower` = `strata-cybertiel:tower-64f17cc2` + serve/ from
  `tower` (TOWER_COMMIT `64f17cc2`).
- Rollback: `polystrata:tower-pr51` is the previous `polystrata:tower` (FROM `strata-cybertiel:pr51`). Run
  `docker tag polystrata:tower-pr51 polystrata:tower`, then `rebuild_container PolyStrata` and `docker start
  PolyStrata` (or Apply in the UI). `Dockerfile.pr51` in the build dir is the old Dockerfile.
- Kept for now, removable: `strata-cybertiel:tower-6ec61d5c` and `polystrata:tower-b36` (PR 36).

### Next

1. (On hold: nothing is posted upstream without the user asking.) Comment on PR 36 with the Tower numbers (Ryzen
   2600X, DDR4-3200: the AMD result it asked for). The microbenchmark gains at 2-4 tokens a call and loses at 1. In the engine on this model there's no
   gain.
2. Done in Pass 14, differently: the engine's calls were counted, the loss came from the gate/up prefetch, and reading
   `down` expert by expert gave the gain without any prefetch.
3. `--skip-weight` on this model: a quality check and more runs, if a few percent of decode is worth changed
   answers.

## Pass 14: PR 36's gain without its 1-token loss (2026-10-19)

Pass 13 left PR 36 at 7-14% faster CPU experts at 2-4 tokens a call and 7% slower at 1, and found no gain in the
engine. This pass first measured what the engine's calls actually look like. It then replayed them in the
microbenchmark, split PR 36 into its parts, and tried other ways to read `down`. **Outcome: a new opt-in,
`STRATA_CPU_DOWN_ORDER=1`, reads `down` expert by expert in tasks of 64 rows, with no prefetch. CPU expert time per
token fell 7-11%, decode rose 1-12%, and 1-token calls got faster too. Tower runs it.** Scripts and sources:
`docs/tower/p14/`, on Tower `/mnt/user/appdata/polystrata/p14/`, results `/data/pass14-*.txt`.

### What the engine's CPU calls look like

A measurement build (`p14/trace.patch` on `tower`: `STRATA_CPU_TRACE=<file>` writes one line per `cpu_experts()`
call from `Engine::moe()`, with its layer, tokens, experts and time, plus a per-request line with the CPU expert ms
per generated token) ran the Pass 12 sequence with `multiturn.py fwd 5`. The result was 234,426 calls and 94 s in
`cpu_experts()` (`p14/tracestat.py`):

| Tokens a call | Calls | Share of the time | µs a call |
|---|---|---|---|
| 1 | 76.2% | 58.4% | 309 |
| 2 | 20.7% | 31.7% | 619 |
| 3-8 | 3.1% | 9.9% | 840-2,890 |

- 95% of the experts in a call have one use, 5% have two. The windows' tokens rarely share an expert that isn't in
  VRAM.
- The MTP draft block (layer 40) is 4% of the time. The main model itself mostly runs 1-token windows on this coding
  traffic: the engine adds guesses only while they pay.
- A 1-token call has 1-8 experts on the CPU (about half of the 8 are in VRAM).

So Pass 13's microbenchmark (all 8 experts on the CPU, 1-4 tokens drawn independently) weighted the wrong shapes.
PR 36 lost 7% exactly where the engine spends 58% of its time.

### Replaying those calls

`p14/cxb.cpp` gained a `trace` mode: the recorded calls (a 60,000-call slice, the same layers, experts and token
counts) run through `cpu_experts()` from the file's mapping, twice after a warm-up pass, timed by call shape. Its
19.2-19.5 s for the slice is close to the 21.2 s the engine recorded for the same calls. The variants are
compile-time switches in `p14/cpu_experts_v.cpp` (today's file when all are off; `p14/build.sh`). All of them gave
the same output hash as today's code. PolyStrata was stopped; Plex, Deluge and the add-on generator ran. Three
interleaved rounds, each variant compared with today's runs in the same round (`p14/benchstat.py`):

| Variant | All calls | 1-token calls | Calls of 9-16 experts |
|---|---|---|---|
| PR 36 (gate/up and down prefetch, down tasks of 32) | +3.4% (-0.2 to +5.8) | +5.4% | -0.8% |
| PR 36's gate/up prefetch alone | +6 to +22% | +6 to +24% | +7 to +18% |
| PR 36's down prefetch alone, down tasks of 16 / 32 | -3.4% / -4.5% (one round disturbed) | | |
| **down by expert ("swap"), tasks of 32** | -8.3% | -6.0% | -12.9% |
| swap, tasks of 32, plus 1 KiB of the next expert's rows asked for ahead | -9.0% | -6.8% | -13.4% |
| swap, tasks of 32, plus all of the next expert's rows asked for ahead | -6.6% | -4.5% | -10.8% |
| **swap, tasks of 64** | **-9.0 to -9.4%** (3 sessions) | **-6.7 to -7.4%** | **-13.3 to -13.7%** |
| swap, tasks of 64, plus 1 KiB ahead | -9.7% (a later session: within noise of swap alone) | -7.7% | -14.2% |
| gate/up: a task reads gate's 32 rows, then up's | ±0.4% | | |
| gate/up: tasks of 64 rows | +0.3% | | |

1. **The gate/up prefetch was PR 36's loss.** A gate/up task reads one expert's 32 rows as one contiguous 36 KiB
   block, which the hardware prefetcher follows. Asking for it again only costs instructions and load slots.
2. **The gain is in `down`'s read order.** Today's down loop goes `for row: for expert`, so each task reads 352
   bytes from each of its experts in turn, as many strided streams as experts. The new order goes `for expert: for
   row` over the task's rows, so each expert's rows of the task are one contiguous read (22 KiB at 64 rows). Every
   product is the same call on the same bytes, and each row's per-use results are kept and summed for each token in
   the same order as before. So the output is the same bits: the same hash at 1, 3 and 4 tokens a call and on the
   replay. With that order, prefetching adds about nothing.
3. **Gate/up changes did nothing**, so gate/up isn't held back by its read pattern here.
4. **Threads:** with the new order, 10 threads are 1.5% and 8 are 2% slower than 12 on the replay (6 is 11%
   slower), as with the old order (Pass 8). 12 stays.

The replay's -9.3% is just under the 10% stop rule set for this pass, but the 1-token calls gain rather than lose,
so it went to the engine.

### The change

`STRATA_CPU_DOWN_ORDER=1` in `src/poly/cpu_experts.cpp` (branch `strata-poly/cpu-down-order` on the fork,
`81d6dfab`, on origin/main `75b79e60`): with the flag, the down products run in tasks of 64 rows, expert by expert,
into a per-thread buffer of 64 x uses floats. Without it, the code path is today's. The engine says
`strata-poly: STRATA_CPU_DOWN_ORDER: CPU experts read down expert by expert` once at load. Built for CUDA (sm_86)
only; the HIP build was not run.

### In the engine

One image (`polystrata:p14ab`: the change plus the per-request CPU-expert line), with the flag on or off, in the
test container with the template's settings, PolyStrata stopped meanwhile. ABBA order, two batches the same day
(`p14/p14ab.sh`, `/data/pass14-ab.txt`, `pass14-ab2.txt`, `p14/abstat.py`). "CPU experts" is the engine's own time in
`cpu_experts()` per generated token over the whole run (it includes the CPU's share of prompt reading, the same
for both). Means, with the range of the runs:

| | Flag off | `STRATA_CPU_DOWN_ORDER=1` | Change |
|---|---|---|---|
| fwd, 4 runs each: CPU experts ms/token | 14.26 (13.08-15.44) | 12.70 (12.25-13.45) | **-10.9%** |
| turn 1, 32K | 62.8 (58.6-65.0) | 68.3 (64.6-71.1) | +8.8% |
| turn 2, 72K | 55.2 (52.6-57.9) | 58.7 (57.6-59.2) | +6.3% |
| turn 3, 103K | 52.2 (50.4-53.7) | 55.2 (53.7-56.6) | +5.7% |
| turn 4, 143K | 47.8 (43.8-49.7) | 51.5 (51.0-52.2) | +7.7% |
| turn 5, 174K | 46.3 (45.3-47.8) | 48.5 (42.5-50.9) | +4.8% |
| rev, 4 runs each: CPU experts ms/token | 15.21 (14.87-15.85) | 13.96 (13.47-14.43) | **-8.2%** |
| turn 1, 35K | 65.4 (63.2-66.7) | 73.5 (67.9-76.4) | +12.4% |
| turn 2, 77K | 56.7 (55.7-57.4) | 59.8 (58.4-61.2) | +5.5% |
| turn 3, 123K | 50.7 (50.2-51.6) | 55.3 (53.1-58.4) | +9.1% |
| turn 4, 204K | 41.7 (40.8-42.6) | 45.0 (44.5-45.3) | +7.9% |
| turn 5, 240K | 38.5 (37.4-40.3) | 39.4 (38.0-40.7) | +2.3% |
| `--window 1` (every call 1 token), fwd 3, 2 runs each: CPU experts ms/token | 11.29 (11.26-11.32) | 10.50 (10.37-10.63) | **-7.0%** |
| turns 1-3, 32K / 72K / 103K | 51.6 / 44.4 / 43.6 | 55.3 / 46.3 / 43.9 | +7.2 / +4.3 / +0.7% |
| short requests (~1,300 tokens of prompt), decode | 54.7-77.5 | 64.7-82.8 | |

1. **CPU expert time per token fell 7-11%,** and decode rose at every turn. The smallest gain is at the longest
   context, where attention takes more of a token.
2. **1-token calls lose nothing; they gain** (the `--window 1` rows: -7.0% CPU time, decode +1-7%).
3. Prompt reading was the same within the runs' spread (1,725-1,886 tok/s at 32K in both). Every 107K recall test
   passed (24 runs). The 247K needle (`TOKENS=185000`, seed 5) passed with the flag on: 3 of 3 facts in 212 s
   (Pass 12: 213 s).
4. Not run: a quality check of answers. The output is bit-identical in the benchmark, and the engine isn't
   deterministic from run to run anyway (Pass 12). Also not run: other models, other CPUs, and HIP.

### What runs now, and the rollback

- `tower` is `5ca982fa`: `strata-poly/cpu-down-order` merged in. Since that branch is on the newest origin/main, the
  merge also brought origin/main up to `75b79e60` (setup, tools, docs and benchmark files only). The engine's
  sources differ from the A/B build only in a comment.
- Images: `strata-cybertiel:tower-5ca982fa` (`p14/Dockerfile.live` on Tower), and `polystrata:tower` FROM it
  (TOWER_COMMIT `5ca982fa`). The template has `STRATA_CPU_DOWN_ORDER=1` (backup
  `my-PolyStrata.xml.pre-cpudown`). Recreated with `rebuild_container PolyStrata` and started: healthy in 6 s, the
  flag's line in the log, 1,987 experts. The Pass 12 sequence: recall passed, decode 66.6 / 60.7 / 54.6 at 32K / 72K /
  103K (Phase 0's base: 63.6 / 54.2 / 49.9).
- Rollback: set the template's `STRATA_CPU_DOWN_ORDER` to 0, or remove it (the flag alone turns it off). For the
  previous image too: `docker tag polystrata:tower-64f17cc2 polystrata:tower` (build dir `Dockerfile.64f17cc2`),
  then `rebuild_container PolyStrata` and `docker start PolyStrata`.
- Measurement images, removable: `polystrata:p14trace`, `strata-cybertiel:p14trace`, `polystrata:p14ab`,
  `strata-cybertiel:p14ab`, and Pass 13's `polystrata:tower-b36` and `strata-cybertiel:tower-6ec61d5c`.

### Next

1. A HIP build of `cpu_experts.cpp` before the branch goes anywhere else. Nothing is posted upstream (the user's
   rule).
2. Other models' expert shapes (GLM 5.3 Flash's down rows are Q2_K/Q3_K, wider experts) were not measured. A
   replay of their traces would show whether 64-row tasks still suit them.

## Pass 15: how long idle CPU workers spin before they sleep (2026-10-19)

`Workers::loop()` in `src/poly/cpu_experts.cpp` spins 20,000 `_mm_pause` before an idle worker sleeps on
`posted_.wait()`. The comment calls that "about a millisecond", which fits Intel since Skylake (a pause is ~140
cycles). The question was whether, on Zen+, the workers fall asleep between layers and pay the wake-up at every call.
**Outcome: they often do, but the wake-ups cost too little to matter. Spinning by time instead
(`STRATA_CPU_SPIN_US`, a measurement build only) gave no visible gain in the engine. Not kept, and the A/B was stopped
early.** Sources: `docs/tower/p15/`, on Tower `/mnt/user/appdata/polystrata/p15/`, results `/data/pass15-*.txt`.

- **Spin length** (`p15/pause.c`, Ryzen 5 2600X): 20,000 pauses take 20-23 µs, 3.6 TSC cycles a pause. So the
  comment is about 50 times too long for this CPU.
- **Measurement build** `polystrata:p15trace` (tower `5ca982fa` + `p15/trace15.patch`): per request, how many posted
  batches found workers asleep, and the time from the post to each worker's first task. It also has the opt-in
  `STRATA_CPU_SPIN_US=N`, which spins N µs by `steady_clock` (read every 256 pauses). With today's spin, in the
  Pass 12 sequence (`multiturn.py fwd 3`, `STRATA_CPU_DOWN_ORDER=1`, `p15/p15ab.sh`):

  | | Batches with sleepers | Workers asleep a batch | Post to first task, woken / spinning | CPU busy during multiturn | CPU experts ms/token |
  |---|---|---|---|---|---|
  | today (20,000 pauses) | 41-53% | 2.3-2.8 of 11 | 18-29 µs / 2.6 µs | 41.8% | 12.18 |
  | `STRATA_CPU_SPIN_US=200` | 15-29% | 0.8-1.7 | 22-33 µs / 2.7 µs | 47.0% | 12.49 |
  | `STRATA_CPU_SPIN_US=1000` | 6-10% | 0.2-0.4 | | 51.1% | 12.24 |

  One run each. Decode at 32K / 72K / 103K: 68.3 / 59.0 / 56.1, 69.6 / 57.2 / 53.1, 62.1 / 56.8 / 53.3, all within
  the spread of Pass 14's runs.
- **Why it can't be much:** a call posts about 2.3 batches (37,427 batches over 400 tokens x 41 layers). About 2.5
  late workers x ~19 µs, shared over 12 threads, is ~4 µs a batch, so at most ~3% of a 309 µs 1-token call. That is
  under the 5% bar. So the gap between a 1-token call's ~25 GB/s and the 44-46 GB/s of sequential reads is mostly
  not wake-up latency.
- **Replay** (`p15/cxb.cpp`, `p15/p15bench.sh`: the Pass 14 trace slice, with `CXB_GAP_US` spinning before each call
  to stand in for the GPU's part of a layer; one round, PolyStrata stopped): no gap 18.17 s; 150 µs gap 20.05 s
  (52% of batches with sleepers), 18.79 s with 200 µs spin (-6.3%), 19.20 s with 1000 µs (-4.3%). The replay sleeps
  more than the engine (4.1 workers a batch against 2.5), so its gain is an upper bound.
- Not run: the ABBA (`p15ab.sh` fwd 5 / rev 5 / `--window 1`). It was stopped in its second run, because one pass
  already showed no real gain. Its partial output is in `/data/pass15-ab.txt`.
- Nothing changed on Tower: PolyStrata runs `polystrata:tower` (5ca982fa) as before. `polystrata:p15trace` and
  `strata-cybertiel:p15trace` can be removed. If the comment matters to anyone, the cheap fix is the comment's wording
  ("about 20 µs on Zen+, a millisecond on Skylake"), not the code.

## Pass 16: parking conversations on the SSD (2026-10-19)

Parking was off on Tower because RAM is too tight for it (the model's files fill the page cache, and the nightly
rsync already pushes experts out). Without it, going back to a conversation after another reads its whole prompt
again: 16.8 s at 32K, 34.5 s at 64K. OpenSpec change `add-poly-ssd-parking`: branch `strata-poly/ssd-parking`
(`df80232b`, from origin/main `974552f5`) adds `--conversation-park-dir DIR`. Parked conversations go to unnamed
(`O_TMPFILE`) direct-I/O files there: their rows in 64 MiB pieces and their last two checkpoints, written by a thread
through a 256 MiB stage. **Outcome: live on Tower. Switching back takes 2.8 s at 32K and 4.7 s at 64K, with no RAM
held and the page cache untouched. The 64K target (under 3 s) is not met.** Results in `/data/ssd-6*.txt`.

- **Tests** (in `strata-cybertiel:tower-4f6270b4`): `glm_conversations_test` passes its RAM run with the same check
  count as upstream (54,950 with libstdc++), then again with a park directory (109,911), then with slow writes so that
  restores race them (164,817), and clean under `-fsanitize=thread`. `poly_park_file_test` passes on `/park` (xfs)
  and on `/var/tmp` (overlayfs). `/mnt/user` (shfs FUSE) is refused: `open(O_TMPFILE | O_DIRECT): Operation not
  supported`. On kernel 6.12, `/dev/shm` (tmpfs) takes the files, so the probe does not refuse it. The headers also
  build with MSVC `/W4 /WX` on the Windows box.
- **Start-time refusals** (exit 2, before the model loads): a missing directory, a path with a space, `/mnt/user`, and
  the flag without `--conversation-cache-mib`.
- **Images:** `strata-cybertiel:tower-4f6270b4` (`p16/Dockerfile.live`, FROM `tower-5ca982fa`, both engines built)
  and `polystrata:tower` = `polystrata:tower-4f6270b4` (TOWER_COMMIT `4f6270b4` = tower `977eeba5` + this branch; since the read-ahead, `polystrata:tower-13061fe1` from `strata-cybertiel:tower-13061fe1`, previous one kept as `polystrata:tower-4f6270b4`).
  The template has a new volume `/mnt/cache/appdata/polystrata/park` -> `/park` (backup `my-PolyStrata.xml.pre-ssd`).
  The engine config has `--conversation-park-dir /park --conversation-cache-mib 8192` (backup
  `strata-cybertiel-ud-q4_k_xl.json.pre-ssd`, which has parking off). The Monitor's `/metrics` gives
  `where: disk, dir: /park`.
- **A/B/A** (`/data/aba.py disk 24000 48000`, `ssd-63-aba.txt`), A's second prompt:

  | | back to A at 32K | back to A at 64K |
  |---|---|---|
  | parking off | 16.8 s | 34.5 s |
  | parking in 2 GiB of RAM | 0.5-0.9 s | 0.4 s |
  | parking on the SSD | 2.8 s (3.7 s wall) | 4.7 s (5.4 s wall) |

  A parked conversation takes 516 MB at 32K and 905 MB at 64K on disk (RAM: 780 MB and 1.17 GB, with all six
  checkpoints). The log's `restored ... in 4426 ms` times all of `enter()`, which first parks the conversation that
  leaves: 0.7 s at 32K and 1.7 s at 64K, the writing that doesn't fit the stage.
- **Where a 64K switch goes** (`/proc/diskstats` of sdg every 0.25 s, `ssd-65-disk.txt`): 1.6 s of writes only at
  ~410 MB/s (B's park, up to the stage), 1.5 s of reads and writes together at ~400 MB/s in all (A's restore while
  B's last 256 MiB is written), then 1.1 s of reads alone at 414-421 MB/s. So the restore's reads run at the drive's
  speed, and the drive's ~410 MB/s, shared by B's 905 MB out and A's 905 MB in, is the limit. `p16`'s
  `readbench.cpp` agrees: 64 MiB direct-I/O reads at 428 MB/s alone, 334 MB/s with 256 MiB of writes queued, writes
  at 409 MB/s.
- **Read-ahead** (`strata-poly/ssd-parking` 39021181, live in tower `13061fe1`): a thread reads the next piece while the engine loads
  one, through two bounce buffers. Same times (`ssd-65-aba.txt`): A2 2.74 s at 32K and 4.65 s at 64K, `enter()`
  2.51 s and 4.43 s (before: 2.54 s and 4.46 s). Loading was never the slow part: a RAM restore of 780 MB takes
  0.29 s. A 6 s pause before A2 (`aba_pause.py`, `ssd-65-pause.txt`) changes nothing either, since B is parked
  when A2 comes.
- **Memory** (`ssd-63-anon.txt`, `RssAnon` of the engine): 1,005 MB with A live before the first park, 608 MB with A
  parked, 625 MB after B parked and A came back, 814 MB at the end with A live again. `Cached:` grew 258 MB while A1's
  prompt was read (model pages), and not at all over the parks and the restore. `df /mnt/cache` grew by the parked
  bytes.
- **SIGKILL** (`ssd-64-kill.txt`): with two conversations parked, `/park` listed nothing, and `/mnt/cache` used
  189,347,389,440 bytes, against 188,183,273,472 before the start. After `kill -9` of the engine: 188,183,265,280, and
  `/park` still empty.
- **Decode** (`decode.py`, `ssd-63-decode.txt`): 68.1, 76.7, 81.0, 75.3 tokens/s (73-79 before).
- **RAM path unchanged** (`ssd-61-old.txt`, `ssd-61-new*.txt`, fixed nonce, `--conversation-cache-mib 2048`): the old
  and the new image log the same `conversation cache:` lines (tokens and bytes) and the same prompt/cached/read counts.
  Their answers differ, but two clean runs of the new image differ too, already at A1, before any parking: greedy
  answers repeat only with `STRATA_FREEZE_EXPERTS=1` (docs/GLM.md). An answer-for-answer check needs that on both.
- **Not tested:** a switch while Plex streams (the shared SSD), `--batch`, and the answer check above.
- **Rollback:** restore the config from `strata-cybertiel-ud-q4_k_xl.json.pre-ssd` and restart. For the previous
  image too: `docker tag polystrata:tower-5ca982fa polystrata:tower`, then `rebuild_container PolyStrata` and
  `docker start PolyStrata`. The `/park` volume can stay.

## Pass 17: where the CPU expert time goes (2026-10-21)

OpenSpec change `add-poly-faster-cpu-experts`. The replay of Pass 14's trace slice (`p17/cxb.cpp`,
`p17/cpu_experts_v.cpp`: the engine's `cpu_experts.cpp` from `strata-poly/cpu-down-order` with the down order always
on) in four builds: **full** (today's code), **arith** (every product reads the call's first expert, so the bytes come
from L2/L3), **reads** (256-bit loads of the same rows in place of the products), **idle** (the tasks return at once).
Every build counts, per worker, the time inside tasks by phase (activations, gate/up, h, down) and call shape (1, 2,
3-8 tokens). A phase's idle share is `1 - busy / (12 threads x its wall time)`. `p17/p17bench.sh`, 3 interleaved
rounds, PolyStrata stopped. Results `/data/pass17-split.txt`.

**Decision rule, written before the 3-round run:**
- Idle share of full at least 10% of the CPU expert time: build the one-batch dataflow (`STRATA_CPU_DATAFLOW`).
- Down arithmetic (arith's down time) at least 70% of full's down time, and down at least 30% of the total: build the
  multi-row Q5_K/Q6_K down product (`STRATA_CPU_DOWN_DOT`).
- Both: dataflow first, then measure again. Neither: the reads take the time; write it up and stop.
- A candidate goes to the engine only if its replay cuts the CPU expert time by 10% or more with the same hash.

**Before the runs.** Tower's kernel marked the TSC unstable at boot (`clocksource: ... 'tsc' skewed 1985153 ns over
watchdog 'hpet' interval of 494018373 ns`) and runs on HPET, so a `steady_clock` read takes 2.5 µs (`p17/clk.c`;
`rdtsc` 32 cycles). The counters read the TSC. The engine doesn't suffer from it: sampling
`/proc/<pid>/task/*/syscall` of `strata-poly` and the server during a 400-token decode found no `clock_gettime` among
about 43,600 samples (futex, poll, read and clock_nanosleep only). The replay's own per-call timer still reads
`steady_clock`, so about 0.3 s of each slice (2 reads x 58,713 calls) is the clock in every build alike.
Counter check (`cxb-full ... idlecheck`): 2,000 batches of 32 tasks of 50 µs on 12 threads measure 50.5 µs a task, and
158-167 µs a batch where 150 is the work: about 8 µs a batch of posting, late starts and waiting. So the idle share
there is 15-18%, not the 11% of the tasks alone. One dry round (`/data/pass17-dry.txt`) ran before this rule was
committed; it is the same rule as in the change's design.md.

**The split** (`/data/pass17-split.txt`, `p17/p17stat.py`; 3 rounds each, PolyStrata stopped, Plex, Deluge and the
add-on generator running). Phase times are summed over the call shapes; idle share weighted by wall time:

| Build | Slice | vs full | act (idle) | gate/up (idle) | h (idle) | down (idle) |
|---|---|---|---|---|---|---|
| full | 17,676 ms | | 281 ms (91%) | 9,626 ms (10.0%) | 211 ms (80%) | 7,318 ms (15.9%) |
| arith (bytes from cache) | 15,218 ms | -13.9% | 278 ms | 7,862 ms | 203 ms | 6,642 ms |
| reads (no products) | 15,518 ms | -12.2% | 221 ms | 9,166 ms | 225 ms | 5,677 ms |
| idle (tasks return at once) | 803 ms | -95.5% | 15 ms | 342 ms | 94 ms | 186 ms |

By call shape, full's idle share is 16.3% at 1 token, 11.8% at 2 and 9.8% at 3-8. Every round of a build gave the same
hash, and full's is Pass 14's swap-64 hash (`188c06d95b5b4137`; 17.63 s against Pass 14's binary at 17.58 s, the same
session).

1. **Gate/up is bound by its reads:** with no products it still takes 95% of its time.
2. **Down is bound by both.** From cache its products take 91% of its time, and its reads alone 78%. They overlap
   well, so making one of them faster leaves the other.
3. **Posting and waiting** (the idle build) is 0.8 s of the slice, 4.5%, about 14 µs a call, and about 0.3 s of that (5 µs a call)
   is the replay's own clock (above).
4. **Rule:** idle share 14.6% (>= 10%), and down's arithmetic 91% of down with down at 41% of the slice. Both, so the
   dataflow went first.

**The dataflow** (`-DCXB_DATAFLOW`, design decision 4: one batch, a down task per expert and block of rows, waiting
for its expert's h; the task that finishes a block's last expert sums each token's products there in its experts'
order). Same hash as full on the slice, and at 1, 3 and 4 tokens in the synthetic mode. Slower
(`/data/pass17-dataflow.txt`):

| Rows a down task | Slice, full / dataflow | Change | Rounds |
|---|---|---|---|
| 64 | 17,752 / 18,409 ms | +3.7% (+3.4 to +4.0%) | 3 |
| 128 | 17,713 / 18,200 ms | +2.7% | 2 |
| 256 | 17,713 / 17,820 ms | +0.6% | 2 |

The idle share overstated the room. The 12 threads are 6 cores with two threads each, so a thread that idles at the end
of a phase gives its core to its sibling, and gate/up is waiting on RAM anyway. What the dataflow adds (4x the down
tasks at 64 rows, workers waiting on an expert's h) costs more than the tails it fills. Not built in the engine.

**The down product, not built.** The reads build is its ceiling: a down product whose arithmetic cost nothing still
waits for the same rows with the same tasks, 5,677 ms against 7,318. That is at most -9.3% of the slice, under the
10% bar before any real kernel. A bit-identical kernel can only share the activation's loads, its `bsums` and the
call itself between rows; every row's own operations stay. A third of the 1,641 ms is about -3%.

**Outcome:** neither path ships. The replay's floor with today's tasks is the reads build, -12% from today, and
nothing that keeps the output bits reaches it. The CPU experts on this machine are close to what its RAM gives at
these call shapes. This ends the search for CPU-side gains on Tower within `cpu_experts()`. What is left there changes
the answers (skipping, lower-bit experts) or moves work off the CPU, which the change ruled out.

Not run: other CPUs (on a CPU with full-width AVX2 and no SMT sharing, the shares differ), GLM's shapes, HIP, and the
engine ABBA (nothing to compare). Binaries on Tower in `/mnt/user/appdata/polystrata/p17/`; nothing changed in the
template or the images.

## Pass 18: PolyStrata 0.2.19-poly.1 (2026-11-22)

Upstream moved from `VecSzn/PolyStrata` to `simplexai-labs/PolyStrata` (the old URL redirects) and went from
`974552f5` (2026-10-19) to `50e42805`, 0.2.19-poly.1 plus one commit: 6,739 commits, most of them the Strata
project's 0.2.16-0.2.19 merges, and a layout move (`src/` -> `engine/src/`, `serve/` -> `terminal/serve/`, setup in
`terminal/`). **Outcome: live since 2026-11-22 on `polystrata:tower` = local branch `tower-next` `c868922e`**
(upstream + our logit_bias, GBNF grammars, constrained JSON and prefill, ported). Not ported, because upstream's own
answer measured better here: SSD parking (`--conversation-park-dir`), `--prompt-cache-first-turn` and
`STRATA_CPU_DOWN_ORDER`. Sources: `docs/tower/p18/`; on Tower `/mnt/user/appdata/polystrata/next/`, results
`/data/upd-*.txt`.

- **Image.** Upstream's Docker image has no strata-poly models, so `p18/Dockerfile` builds one: CUDA 13.0 devel,
  the repo from `git archive`, `terminal/.venv`, the sampler and grammar tests, then `setup.build_engine_poly` for
  sm_86 with the CPU image encoder. It starts `terminal/serve/server.py` with `/data/config/strata-cybertiel-next.json`
  (the old config less `--conversation-park-dir` and `--prompt-cache-first-turn`, paths under
  `/opt/strata/terminal`). No entrypoint: the template's FAMILY/MODEL/... variables are gone. The image has no `curl`.
- **Template** (`my-PolyStrata.xml`, backup `.pre-0219`): env `TMPDIR=/park` (upstream spills parked conversations to
  nameless files in `$TMPDIR`) and `STRATA_POLY_MIRROR=0`; FATTN/FDEC are upstream defaults now (on from 4,096 tokens).
- **Numbers** (warm, same prompts: the old image's `docs/` and `README.md` copied in for `long_prompt`):

  | | live 0.1.45 (`tower-893468a5`) | 0.2.19 + ours, mirror off |
  |---|---|---|
  | decode, 400 tokens after a short prompt | 64-72 tok/s | 77-88 |
  | read 11K | 7.4-8.1 s | 5.8-6.0 s |
  | read 32K | 17.7-18.0 s | 16.5-16.8 s |
  | read 64K | 35.7-36.4 s | **40.7-42.2 s** (slower: no flash-attention prompt kernel, Pass 19) |
  | back to A after B, 32K / 64K | 3.6 / 5.0 s | 0.5-0.9 / 1.0-1.4 s |
  | add-on sim (`addon_sim.py`): forge prompt after the first, median | 4.45 s (81.5% reused) | 3.04 s (70.8% reused) |
  | engine RssAnon after the set | 0.9 GB | 2.7 GB (17 GB with the mirror) |

- **The mirror.** 0.2.19 copies the experts outside VRAM into pinned RAM at load (14.7 GiB here, 60 s load). On
  Tower it bought nothing: decode, 11K/32K/64K reads and switches were the same within noise with
  `STRATA_POLY_MIRROR=0`, which loads in 20-24 s and leaves 25.7 GB available against 10.7. Off.
- **Threads.** The default is now 70% of the CPUs (8 of 12). `--threads 12` gave the same numbers, with and without
  the mirror. Default kept.
- **The 64K read** is slower in every variant (mirror, threads); `STRATA_GDN_MMA=0` made it slower still (44.8 s).
  Cause and fix in Pass 19.
- **Add-on pattern.** Upstream keeps a checkpoint where the last turn starts and every 2,048 tokens; a new forge
  request went on from 12,288 of ~13.5K system tokens. Live went on from all 14,149 (first-turn checkpoint) but its
  SSD restore cost more than the 1.3K tokens it saved. `"warm_start": "agent.json"` with the forge system prompt
  would pin it whole (not tried).
- **API checks** (`p18/features.py`): `/v1/status` says `grammar_and_validate`; a yes/no grammar, a json_schema mob,
  prefill on both APIs, and a -100 bias on `Hello` (9419 -> `Greetings`) all pass on the live container.
- **Rollback:** `docker tag polystrata:tower-893468a5 polystrata:tower`, copy `my-PolyStrata.xml.pre-0219` over the
  template, `rebuild_container PolyStrata`, `docker start PolyStrata`. The old config is untouched.

## Pass 19: why 0.2.19 reads 64K slower (2026-11-22)

Pass 18 left one regression: a cold 64,620-token prompt took 40.7-42.2 s on 0.2.19 against 35.7-36.4 s on 0.1.45,
while 11K and 32K got faster. **Cause: the 0.2.19 engine has no flash-attention prompt kernel.** Upstream now builds
strata-poly without ggml by default (`STRATA_POLY_MMQ` off; the build log says `Strata: no ggml in this build (own CPU
pool, no MMQ, tests off)`). `fattn.cu` is then compiled with `STRATA_POLY_OWN`, so `fattn_fits()` is false. A prompt's
attention goes through `attend_batch()`, and the engine says nothing unless `STRATA_FATTN=1` is set. With that set it
says `prompt attention unchanged (no kernel for it on this card ...)`. 0.1.45 linked llama.cpp's ggml-cuda and used the
kernel. **Fix: build with the vendored llama.cpp tree**, `-DSTRATA_GGML_DIR=/opt/strata/engine/third_party/strata-llama
-DSTRATA_POLY_MMQ=ON` (no download). With that build 64K reads in 28.0-28.8 s and 32K in 13.3-13.4 s. Image
`polystrata:p19-mmq` (`p19/Dockerfile`, on top of `next-c868922e`, 5 minutes). It is not live: the template and the
`polystrata:tower` tag are unchanged. Sources: `docs/tower/p19/`; on Tower `/mnt/user/appdata/polystrata/p19/`.

**How it was measured.** These runs used the test container on port 5003, with the live container stopped and the
mirror off. Pass 18's settings: `--kv q8_0`, 262,144 context, 8 threads. The cold reads are `aba.py`'s A1 and B1 at
sizes 24000 (32,113 tokens) and 48000 (64,620). For a per-layer split, `p19/pplog.py` patches the test container's
server to log each `PP` progress line the engine sends, with its time. The engine sends one at least every 500 ms
during a read, and a read goes layer by layer over the whole prompt. At 64K a layer takes over 500 ms, so most lines
span exactly one layer of the 41 turns (40 layers and the draft block). `p19/split.py` averages those lines: the 10
full-attention layers (3, 7, ... 39) and the GDN layers. Layer 0 and the last turn are left out.

**Settings tried on 0.2.19** (`p19/all1.sh`, one round each, cold reads in seconds):

| 0.2.19 as live, and | 32K | 64K |
|---|---|---|
| nothing (two rounds, an hour apart) | 16.5-16.7 | 40.6-41.7 |
| `STRATA_FATTN=0` | 16.5-16.6 | 40.7-41.8 (no kernel to turn off) |
| `--prompt-cache-every 0` | 16.4-16.5 | 40.6-40.8 |
| `--prompt-cache 0` | 16.1-16.3 | 40.5 |

The checkpoints are not the cause. Upstream's `8acd31d0e` (checkpoints copied aside during a read) changes only the
path where every expert is in VRAM. This card reads layer by layer, as both builds do at 64K. 0.1.45's 512 MiB
residual limit holds 65,536 tokens, so it is one read there too.

**Per layer at 64K** (ms, mean of the one-layer lines, both cold reads of a round):

| Build | Cold read 64K | Attention layer (x10) | GDN layer (x29) |
|---|---|---|---|
| 0.1.45, FATTN on (as live was) | 35.0-36.2 s | 1,217-1,220 | 727-730 |
| 0.1.45, `STRATA_FATTN=0` | 48.6-49.9 s | 2,574-2,579 | 728-730 |
| 0.2.19 as live | 40.8-41.3 s | 2,424 | 525-530 |
| 0.2.19 with ggml (`p19-mmq`) | 28.0-28.7 s | 1,152-1,154 | 522-526 |

- 0.2.19's GDN layers are 28% faster than 0.1.45's (727 -> 525 ms). Over 29 layers that saves about 6 s at 64K.
- Its attention layers take what 0.1.45's took with the kernel off (2,424 against 2,574 ms), twice their time with it
  on. Over 10 layers that costs about 12 s. Net: about +6 s.
- At 32K the attention layers take 745 ms on 0.2.19, against 287 ms for a GDN layer: a smaller share, so the GDN gain
  wins there (32K: 0.1.45 16.9-17.3 s this session, 0.2.19 16.5-16.7 s).
- llama.cpp's MMQ in place of Strata's own prompt products leaves the GDN layers where they were (522-526 ms). All
  of the gain is the attention kernel.

**The fixed build against today's** (`p19/nx.sh`, two rounds each, interleaved: base, mmq, base, mmq):

| | 0.2.19 as live | `p19-mmq` |
|---|---|---|
| read 11K (`bench_11k.py`, 3 runs, one round) | 5.93-5.94 s | 5.57-5.61 s |
| read 32K | 16.5-16.7 s | 13.3-13.4 s |
| read 64K | 40.6-41.7 s | 28.0-28.8 s |
| decode, 400 tokens after a short prompt (runs 2-4 of 4; one round / two) | 73.3-76.8 tok/s | 75.0-87.2 |

The first decode run after a load is slower in both (55.5 and 57.0 tok/s) and is left out, as in Pass 18.

**A second kernel that is off: FDEC.** The tensor-core decode kernel is off in both 0.2.19 builds, for a different
reason. Its keys now go in splits of 256 (`kSplit`, 1,024 in 0.1.45). Its scratch at 262,144 positions is
1,024 splits x 2 KV heads x 32 rows x 258 floats = 16.9M floats. The engine gives it `chunk_` (4) x 16 heads x
262,144 = 16.8M floats, so `fdec_fits()` is false (`STRATA_FDEC=1` prints `no kernel for it on this card`). It fits at
`--max-context` 209,715 or less, where `chunk_` is 5. This was not measured. FDEC starts from 4,096 positions, so
`decode.py`'s short prompt never reaches it. The cost shows in long-context decode.

**Not checked:** whether the two 0.2.19 builds give the same answers token for token (the kernels round
differently), GLM's models, and the add-on pattern on `p19-mmq`.

**Not taken (2026-11-23).** Strata has moved away from llama.cpp, so Tower stays on upstream's default build.
Upstream put the prompt kernel back in that build the next day (Pass 20). What follows is kept for the record.
**To make it live** (not done): `docker tag polystrata:p19-mmq polystrata:tower`, then `rebuild_container PolyStrata`,
`docker start PolyStrata`. The template already names `polystrata:tower`. Rollback: tag
`polystrata:tower-c868922e` back. For the next image build, add the two `-D` options to `p18/Dockerfile`'s engine step.
`setup.build_engine_poly` passes `STRATA_GGML_DIR` from the environment but has no switch for `STRATA_POLY_MMQ`, so
the build has to call CMake itself, as `p19/Dockerfile` does.

## Pass 20: PolyStrata 0.2.20-poly.4-hotfix (2026-11-23)

Upstream fixed both kernels Pass 19 found off, in the build without llama.cpp, in 0.2.20-poly.2:

- `113e80ace`: the prompt flash-attention kernel cut down to the one case strata-poly launches (heads of 256, Ampere
  or later), with no llama.cpp header. Found from issue 263 (an RTX 3080's 122K read 29% slower than on 0.1.45).
- `cc02708a1`: the decode kernel (FDEC) takes 1,024 keys a block where 256 do not fit the scratch. The same shortfall
  as Pass 19 (16,908,288 floats needed, 16,777,216 there at 262,144 positions).

**The branch.** `tower-next` `aadae31ea` merges `v0.2.20-poly.4-hotfix` into `4da66d8cb` (our logit_bias, grammar,
constrained JSON and prefill ports). Two conflicts: `engine.cpp` keeps both includes; `server.py` keeps the grammar
block and both new flags (`--constrained-decoding`, upstream's `--strict-model`). Upstream's new second `prepare()` on
the Anthropic path (#6460, unasked thinking turned off near the context limit) passes `prefill` as the first one does.
Image `polystrata:next-aadae31e`, built from `next/` as before (the Dockerfile is Pass 18's). Its engine log says
`STRATA_FATTN: prompt attention through llama.cpp's flash-attention kernel` and `STRATA_FDEC: decode attention
through the tensor-core decode kernel from 4,096 tokens of context`. Live from 2026-11-23 with `STRATA_CPU_NO_X4=1` (below).

**How it was measured.** `p20/t.sh` runs the test container with live's environment and config; `p20/chain.sh` ran
new, current (`tower-c868922e`), new, current. Each run: `features.py`, the three ported features' unit tests inside the
image, `runbench.sh` (decode, `bench_11k.py`, `aba.py` 32K and 64K, RSS) and `p20/ldecode.py` (300 tokens with
thinking off, after 11K, 32K and 64K). Then `p20/pchain.sh` ran `p20/dprobe.py` (decode with draft acceptance, thinking
on: 400 tokens after a short prompt x4, 300 after 11K x2) on new, current, and new with `STRATA_FDEC=0`.

| | 0.2.19 as live (`c868922e`) | 0.2.20 hotfix (`aadae31e`) |
|---|---|---|
| `features.py`; grammar, logit_bias and prefill unit tests | all ok; 46 ok | all ok; 46 ok |
| read 11K (runs 1-3, both rounds) | 5.95-6.24 s | 5.64-6.03 s (one 7.40, first after load) |
| read 32K (A1, B1) | 17.0-17.3 s | 13.7-14.6 s |
| read 64K (A1, B1) | 41.8-43.7 s | 29.1-30.0 s |
| follow-ups on a cached 32K or 64K prompt | 0.4-1.6 s | 0.4-1.6 s |
| decode after a short prompt, `runbench.sh` (runs 2-4) | 79.8-87.0 tok/s | 67.0-74.3 |
| decode after a short prompt, `dprobe.py` (4 runs; mean) | 78.7-85.8 (83.0) | 72.3-80.6 (75.7) |
| draft acceptance there | 73-86% | 79-83% |
| decode after 11K, thinking on (`dprobe.py`) | 75.6-79.7 | 75.5-76.7 |
| decode after 11K, thinking off (`ldecode.py`) | 60.9-65.5 | 41.6-49.7 |
| decode after 32K, thinking off | 56.0-63.2 | 57.5-59.9 |
| decode after 64K, thinking off | 48.5-54.0 | 57.4-60.1 |
| engine RSS after the set | 18.4-19.2 GB | 17.7-18.6 GB |

- Prompt reads: 64K 30% faster, 32K 17% faster, and 64K is now 6 s under 0.1.45's 35-36 s.
- Decode at 64K: 11% faster on the mean, the decode kernel's work.
- **Decode after a short prompt is 9% slower** (dprobe means 83.0 -> 75.7 tok/s, about 1.1 ms more a token). Draft
  acceptance is the same, and `STRATA_FDEC=0` leaves it where it is (72.3-80.5, mean 75.8), so it is neither the
  drafts nor the decode kernel (which starts at 4,096 positions anyway). Not found yet. Upstream engine commits on
  this path since 0.2.19 include `5d25761c1` (upload order), `e7f467548` (four weight rows a call in the CPU experts'
  products), `034399e2a` (cached experts' offsets sent by a kernel) and the Strata 0.2.20 merge (`d28c053e6`, 8 GB and
  12 GB cards).
- After 11K the two scripts disagree: equal with thinking on, 26% slower with thinking off. `ldecode.py` records no
  draft acceptance, and one 300-token answer a run is noisy. Not settled.

**Against 0.1.45, the baseline** (the image Tower ran before Pass 18, `tower-893468a5`, with its template's
`STRATA_FATTN=1`, `STRATA_FDEC=1`, `STRATA_CPU_DOWN_ORDER=1`; `p20/o.sh`). `p20/chain2.sh` ran 0.1.45, new, 0.1.45, new,
the same set as above plus `dprobe.py` after `ldecode.py`. Both 0.1.45 runs logged all three switches on.

| | 0.1.45 | 0.2.20 hotfix | |
|---|---|---|---|
| `features.py` | all ok | all ok | |
| read 11K (runs 1-3) | 7.65-8.01 s | 5.56-5.76 s | -28% |
| read 32K (A1, B1) | 17.8-18.1 s | 13.8-13.9 s | -23% |
| read 64K (A1, B1) | 35.6-36.6 s | 28.3-29.2 s | -20% |
| back to A after B, 32K / 64K (A2) | 3.0 / 4.9-5.0 s | 0.6-0.8 / 0.9-1.6 s | |
| decode after a short prompt, `runbench.sh` (runs 2-4; mean) | 64.4-79.1 (74.8) | 71.3-77.3 (73.4) | -2% |
| decode after a short prompt, `dprobe.py` (runs 2-4; mean) | 75.0-79.9 (77.4) | 60.3-70.8 (67.7) | -13% |
| draft acceptance there | 77-85% | 79-89% | |
| decode after 11K, thinking on (`dprobe.py`) | 72.9-77.2 | 68.3-72.1 | -6% |
| decode after 11K, thinking off (`ldecode.py`) | 65.9-66.6 | 48.7-52.8 | -23% |
| decode after 32K, thinking off | 72.7-74.9 | 57.8-63.3 | -17% |
| decode after 64K, thinking off | 70.8-72.9 | 55.2-60.5 | -20% |
| engine RSS after the set | 23.0-23.7 GB | 18.8-19.6 GB | |

- Every prompt read is faster, and going back to a conversation after another one is 3-5 times faster.
- **Decode is slower than 0.1.45**, most after a long prompt (17-23% at 11K-64K with thinking off). Draft acceptance is
  the same, so each step costs more: about 14-15 ms a token against 12.5-13.3. The 0.2.19 image is slower still after
  64K (48.5-54.0). The decode kernel's fix recovers part of what 0.2.x lost there, not all of it.
- `dprobe.py`'s short decode is lower on the new image when it runs after `ldecode.py`'s long prompts (60.3-70.8) than
  when it runs first (72.3-80.6, the probe above). 0.1.45 does not show that.
- The 0.1.45 image's own copies of the grammar, logit_bias and prefill unit tests fail 2 of 46; its `features.py` run is
  clean. Its tests are not the ones the new image runs.

**Cause of the decode loss: the CPU experts' four-row kernels (`e7f467548`).** Found with the engine alone
(one-shot mode, `p20/cli/`), not the server: the same token ids for both builds (`gen_ids.py`: the INI question,
33 tokens, and the same after an 11K `long_prompt`, thinking off), 300 tokens, each run a fresh load with its own copy
of the expert profile. The engine's last line gives decode tok/s, the share of experts from VRAM and CPU expert ms a
token. The first run after the live container stops is always slow (17.8-46.2 tok/s) and is left out.

- 0.2.20 against 0.1.45 (`run1.sh`): short 59.3-59.6 against 70.9-72.9 tok/s, 11K 52.9-54.4 against 63.7-64.2. The
  new engine takes more experts from VRAM (62-67% against 54-60%) and still spends more CPU time a token (12.0-12.7
  against 9.6-10.0 ms): each CPU expert costs about 1.5 times as much.
- Not the threads (`run2.sh`): 0.2.x takes 70% of the CPUs by default (8 of the 2600X's 12; 0.1.45 took 12), but
  `--threads 12` gives 57.6 short and 53.4-55.1 at 11K, `--threads 10` 62.6 and 53.5.
- Not the mirror: `STRATA_POLY_MIRROR` on, 62.7 short.
- Between 0.2.19 and 0.2.20 (`run3.sh`): 0.2.19 72.0 short (CPU 9.7 ms) and 59.6-61.2 at 11K; 0.2.20 60.4-61.4
  (12.1-12.3 ms) and 50.6-54.2. 0.2.20's one CPU-expert change is `e7f467548`: a gate, up or down matrix four weight
  rows a call where the quant type has a four-row kernel (Strata 0.2.20's `dot_x4_kernel()`). Upstream measured it on
  an i7-14700HX at +2% with every expert on the CPU and level with a split. It has no switch.
- `p20/cli/x4.patch` adds one, `STRATA_CPU_NO_X4=1` (one row a call, as before), image `polystrata:p20-x4` on top of
  `next-aadae31e` (`p20/cli/Dockerfile`). `run4.sh`, after a discarded warm-up, two rounds in turn:

| Engine alone, tok/s | short | CPU experts ms a token | after 11K |
|---|---|---|---|
| 0.2.20, four-row (upstream default) | 58.7-59.5 | 12.1-12.8 | 51.9-54.6 |
| 0.2.20, `STRATA_CPU_NO_X4=1` | 69.9-71.1 | 9.8-10.0 | 60.5-63.4 |
| 0.2.19 | 67.0-68.2 | 10.0 | 59.7-60.3 |
| 0.1.45 (FATTN, FDEC, CPU_DOWN_ORDER on) | 68.0-73.7 | 9.5-10.2 | 65.1-67.9 |

  With the switch, short decode is back to 0.1.45 and 0.2.19; after 11K it is 0.2.19's and 4-7% under 0.1.45.
  `STRATA_CPU_DOWN_ORDER=0` on 0.1.45 cost about that much there (64.2 -> 60.7, `run1.sh`), so the rest is likely
  Tower's own CPU-expert change from Pass 14, which 0.2.x does not have. Not measured on 0.2.20. The four-row kernels
  are likely slow on the 2600X because Zen+ runs AVX2 on 128-bit units; this was not profiled.

**The server set with the switch.** `tower-next` `9172f75e8` carries it (opt-in; the default path is upstream's).
Image `polystrata:next-9172f75e`. `p20/chain3.sh` built it and ran 0.1.45 (`o.sh`), the new image with
`STRATA_CPU_NO_X4=1` (`t.sh`, which now takes environment settings), 0.1.45, new.

| | 0.1.45 | 0.2.20 + `STRATA_CPU_NO_X4=1` | |
|---|---|---|---|
| `features.py`; the image's grammar, logit_bias, prefill tests | all ok; 2 of 46 fail (its own copies) | all ok; 46 ok | |
| read 11K (runs 1-3) | 7.31-7.50 s | 5.52-5.58 s | -25% |
| read 32K (A1, B1) | 17.3-17.8 s | 13.4-13.9 s | -22% |
| read 64K (A1, B1) | 35.3-36.6 s | 28.2-28.7 s | -21% |
| back to A after B, 32K / 64K (A2) | 2.9-3.0 / 4.9-5.0 s | 0.7 / 0.9 s | |
| decode after a short prompt, `runbench.sh` (runs 2-4; mean) | 75.7-82.1 (79.2) | 77.5-87.9 (82.4) | +4% |
| decode after a short prompt, `dprobe.py` (runs 2-4; mean) | 72.2-80.6 (77.6) | 60.6-84.7 (74.6) | -4% |
| decode after 11K, thinking on (`dprobe.py`) | 75.9-80.2 | 81.2-84.1 | +7% |
| decode after 11K, thinking off (`ldecode.py`) | 67.0-68.5 | 64.4-66.1 | -4% |
| decode after 32K, thinking off | 74.8-76.4 | 68.2-71.6 | -7% |
| decode after 64K, thinking off | 70.0-72.1 | 62.7-63.8 | -11% |
| engine RSS after the set | 24.6-24.7 GB | 19.9 GB | |

- Short-prompt decode is level with 0.1.45, within the runs' spread. Without the switch it was 9-13% under.
- Decode after a long prompt is still under 0.1.45, more so the longer the prompt: -4% at 11K, -7% at 32K, -11% at
  64K (without the switch it was -23%, -17%, -20%). Not found yet. Candidates: Tower's `STRATA_CPU_DOWN_ORDER`
  (about 5% after 11K on 0.1.45, and not in 0.2.x), and the decode attention at long context (0.2.20's kernel takes
  256 keys a block where its scratch holds them, 0.1.45's 1,024).
- `features.py` and the unit tests passed on every run; the engine logged no errors.

**Not checked:** GLM's models, the add-on pattern, a long agent session.

**Live since 2026-11-23 20:50.** `polystrata:tower` = `polystrata:tower-9172f75e` = `next-9172f75e`. The template
(`my-PolyStrata.xml`, backup `.pre-0220`) gains `STRATA_CPU_NO_X4=1` beside `TMPDIR=/park` and `STRATA_POLY_MIRROR=0`.
After `rebuild_container PolyStrata`, the engine process has the switch in its environment. It logs both kernels on and
loads in 19.7 s with 2,287 experts in VRAM. `features.py` is all ok, and `decode.py` gives 67.4, 80.1, 84.2 tok/s (the
first run after a load is the slow one). The decode gap after long prompts (-4% to -11% against 0.1.45) stays open.
Rollback: `docker tag polystrata:tower-c868922e polystrata:tower`, restore `my-PolyStrata.xml.pre-0220`,
`rebuild_container PolyStrata`, `docker start PolyStrata`. For 0.1.45, use `tower-893468a5` and `.pre-0219`.

## Pass 21: Inkling 2 Mini on the live image (2026-11-23)

Inkling 2 Mini (Thinking Machines, Unsloth's GGUF; upstream's notes are `docs/models/inkling.md`) on live's image
`polystrata:tower-9172f75e`, which already has upstream's Inkling 2 engine (`inkling2_loop`, the small-card settings,
`--cache-bias`). Run in the test container only. Live stayed on Cyber-Tiel-Coder.

**The file.** UD-Q4_K_XL, the largest size setup offers. Setup's RAM estimate for it beside an 8 GB card is 21 GB, and
Tower has 31 GiB. Upstream has not run this size on an 8 GB card. Both files come from `unsloth/Inkling-2-Mini-GGUF` at
setup's pinned revision `ea0c79b2`: `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` (17,629,760,736 bytes) and
the picture reader `mmproj-F16.gguf` (639,348,160 bytes). They are in `/mnt/cache/llm/inkling/`, apart from the Qwen
`mmproj-F16.gguf` of the same name. Both sha256 sums match setup.py's. One curl got about 2 MB/s, so `p21/pdl.sh`
fetches 16 byte ranges at once and joins them. Both files took about 13 minutes, sha256 checks included.

**The config.** `/data/config/strata-inkling2-mini.json` (`p21/strata-inkling2-mini.json`), written by hand the way
`setup.py` writes an Inkling 2 config on this card:
- 64K context.
- `--conversation-cache-mib auto`.
- The three small-card settings that setup adds when the card can't hold every expert: `--attn-requant q5_0`,
  `--loop-bf16`, `--tail-skip 4`.
- Pictures read on the CPU (6 threads, 300 tokens).
- Live's `--vram-reserve-mib 512` and `"constrained_decoding": true`.

The tokenizer pack is `/data/packs/inkling-ud-q4_k_xl`, built by the image's `tools/strata_tokenizer.py --gguf`
(vocab 200,058, pretokenizer `inkling2`). The engine writes the expert profile there when it stops. Live's environment:
`TMPDIR=/park`, `STRATA_POLY_MIRROR=0`, `STRATA_CPU_NO_X4=1`.

**How it was measured.** `p21/t.sh` is `p20/t.sh` with the Inkling config (`CFG=` picks another config). It stops live
`PolyStrata` first. Each round loads the model, then runs:
- `p21/features.py`: `/data/features.py` with the tokenizer path as an argument, for logit_bias.
- `p21/stream.py`: streaming on both APIs, with thinking on and off.
- The image's grammar, logit_bias and prefill unit tests.
- `decode.py` (400 tokens x4).
- `bench_11k.py`.
- `aba.py 24000`, about 32K tokens.
- `ldecode.py 8000 24000`: 300 tokens with thinking off, after about 11K and 32K.

`p21/chain.sh` ran a second round of setup's config (b), then a round without the three small-card settings (c,
`p21/acc.json`, upstream's high-accuracy mode `--inkling-small-card off`). Round a was the first load. Its stream test
was later fixed to ask the Anthropic API for no thinking explicitly (below), and was rerun on the same load. The
prompts are counted in Inkling's tokens: `bench_11k.py` makes 10,872 and `aba.py 24000` makes 32,702.

| Inkling 2 Mini UD-Q4_K_XL, RTX 3070 8 GB | setup's config (a, b) | without the small-card settings (c) |
|---|---|---|
| load | 20.3-21.2 s | 19.1 s |
| experts in VRAM | 997 and 1,103 of 3,072 (4.8 GiB) | 1,043 (4.5 GiB) |
| `features.py` (grammar, json_schema, prefill both APIs, logit_bias) | all ok | all ok |
| streaming, both APIs, thinking on and off | all ok | all ok |
| grammar, logit_bias and prefill unit tests | 46 ok (1 skipped) | 46 ok (1 skipped) |
| decode after a short prompt (runs 2-4) | 53.6-57.7 tok/s | 37.6-38.2 |
| decode after 10,956 tokens, thinking off | 42.9-45.6 | 34.4 |
| decode after 32,745 tokens, thinking off | 39.6-41.3 | 30.4 |
| read 10,872 tokens (runs 1-3) | 16.8-17.9 s (607-648 tok/s) | 16.6-17.0 s |
| read 32,702 tokens (A1, B1) | 42.0-45.8 s (714-778 tok/s) | 41.2-43.0 s |
| back to A after B at 32K (A2) | 1.5-1.7 s | 2.1 s |
| engine RSS after the set | 14.7-14.9 GB (13.0 GB of it file pages) | 15.2 GB |

- **It runs and answers on both APIs, streamed and not, with thinking on and off.** The engine logged no errors.
- **The small-card settings are worth 29-46% in decode on this card**: 55 against 38 tok/s short (+46%), and
  41 against 30 after 32K (+33%). They don't change prompt reads. Upstream measured what they cost in answers on other cards: the same most
  likely token at 94.3-95.0% of steps with all four (it measured `--expert-swaps 0` too), against 98.2-99.0% for a
  second run without them. `--tail-skip 4` leaves out about a quarter of a token's expert picks on 8 GB cards, where
  upstream didn't measure what that does to answers. Tower didn't measure the answers either.
- **Against Cyber-Tiel-Coder on the same image and card** (Pass 20's last table, the column with the switch):
  - Inkling reads prompts about 3 times slower: 11K in 16.8-17.9 s against 5.5-5.6 s, 32K in 42.0-45.8 s against
    13.4-13.9 s.
  - Inkling decodes at a half to two thirds of Cyber-Tiel's speed: 54-58 against 78-88 tok/s short, 40-41 against 68-72 after 32K.
    Cyber-Tiel guesses ahead with its MTP head, and Inkling has no draft model here.
  - The card holds a third of Inkling's experts. Its 24 layers run as 48, so a prompt batch sends the rest over
    PCIe 3.0 for every layer run.
  - Not checked: `--prompt-batch 2048`. setup.py says it halves how often a small card copies the missing experts,
    but setup doesn't give it to Inkling.
- 0.1.45 has no Inkling 2 engine, so there is no baseline run.
- **Pictures** (`p21/vision.py`: a red circle and the text "TOWER 21", 448x448): both APIs answer "The shape is red,
  and the text says "TOWER 21"." in 6.2 s with the CPU reader (179 prompt tokens). The same picture again reads from
  the cache in 0.4 s.

**Our ported features with Inkling 2.**
- Grammar, constrained `json_schema` and logit_bias: work. The answers are `no`, `{"name":"Creeper","hp":20}`, and
  `Hello` -> `Hi` with token 32269 banned.
- **Prefill: the continuation is right, but it ends with the literal text `<|end_message|>`** on both APIs (`' three,
  four, five<|end_message|>'`). `features.py` passes it because it only looks for "three". A plain answer ends cleanly.
- The prefill cause is in our port (`c868922e3`, `terminal/serve/server.py`), not in upstream, which has no prefill:
  - `Service.run` tells Inkling's output parser that the reply's text block is open only when the prompt ends with
    `<|message_model|><|content_text|>` (`opened = ids[-len(self.text_open_ids):] == self.text_open_ids`).
  - `prepare()` adds the prefill's tokens after that opening.
  - So the parser starts outside a block and passes the block's end through as text.
  - Fixed later in this pass (`f6bc1acf1`, below).
- **Thinking is on unless the request turns it off, on both APIs.** An Anthropic request with no `thinking` field
  thinks, as an OpenAI request without `enable_thinking: false` does. With `max_tokens` 32 that left no text.
  `"thinking": {"type": "disabled"}` gives the answer straight away.

**Not checked:** tool calls, a long agent session, `"parallel"`, `--kv fp8` (about 0.7 GB more for experts, per
upstream), `--cache-bias`, contexts past 32K, and how far the small-card settings move the answers on this card.
After the runs: `docker rm -f polystrata-next && docker start PolyStrata`. Live is back on Cyber-Tiel-Coder.

**Offered as a second container (2026-11-23).** `PolyStrata-Inkling`, from the template
`/boot/config/plugins/dockerMan/templates-user/my-PolyStrata-Inkling.xml` (`p21/my-PolyStrata-Inkling.xml`). It is
live's template with:
- Port 5004.
- `PostArgs` that start the server on `/data/config/strata-inkling2-mini.json` (in place of the image's `CMD`, which
  names Cyber-Tiel's config).
- Its own name and description.

It has the same image (`polystrata:tower`), mounts and environment as live, so it follows live's image when that
container is rebuilt. Live's template was not changed. Neither container is set to autostart.

The two share the one card, so only one runs at a time:

```sh
docker stop PolyStrata && docker start PolyStrata-Inkling     # Inkling 2 Mini on :5004, loads in about 25 s
docker stop PolyStrata-Inkling && docker start PolyStrata     # back to Cyber-Tiel-Coder on :5002
```

Created with `rebuild_container PolyStrata-Inkling` while live was stopped. That command starts a new container
once and stops it again if it isn't set to autostart. Then `docker start PolyStrata-Inkling`:
- It loaded in 21.1 s with 1,103 experts in VRAM.
- `features.py`, `stream.py` and `vision.py` passed, with prefill's `<|end_message|>` as above.
- `decode.py` gave 44.0, 49.2 and 52.0 tok/s, the first runs after a load.
- `/v1/models` answers from the LAN at `192.168.1.166:5004`.

Then it was stopped and `PolyStrata` started again. `features.py` passes on it.

**The prefill fix.** The fix is commit `f6bc1acf1` on `tower-next`. It adds `open_block_at_end()` to
`terminal/serve/server.py`, which reads back from the prompt's end. If the text block's opening comes before any
`<|end_message|>`, the reply's text block is open, with or without a prefill after it. An earlier reply in the history
always ends with `<|end_message|>`, so it never counts as open. Only Inkling 2 sets `text_open_ids`, so other
families' path is the same as before.

`test_inkling.py` gains two tests:
- A prefill after the opening, with one and with two user turns.
- A prompt that ends after a closed text block.

The first failed before the fix with `'three<|end_message|>'`. The tests ran in the image on Tower with this `serve/`
mounted over its own: grammar, Inkling, logit_bias, prefill, run config, server and structured output, plus the family
parsers. 601 tests ran, and one failed: `test_server.LoadSharedSettings.test_several_bad_keys_are_each_named`, which
also fails without the fix (599 tests there). `unittest discover` over all of `serve/` hung with the CPU idle and was
stopped after 10 minutes. Not looked into.

Image `polystrata:next-f6bc1acf`, built from `next/` as before. `next/repo` is the new commit's archive, and the old
one is `next/repo-9172f75e`. They differ only in `server.py` and `test_inkling.py`. `p21/chain2.sh` ran on that image:

- The Inkling set (`t.sh fix`): prefill is `' three, four, five'` on both APIs, without the tag.
  - Everything else is as before.
  - Decode 56.1-58.4 tok/s short, 42.8 after 11K, 40.7 after 32K.
  - Reads 17.4-17.6 s at 11K and 42.5-46.0 s at 32K.
- Cyber-Tiel's live config: `features.py` all ok. The image's grammar, logit_bias, prefill and Inkling tests: 100 ok
  (4 skipped).

Then live started again, still on `polystrata:tower` = `tower-9172f75e`. `features.py` passes on it. Live and
`PolyStrata-Inkling` get the fix only when `polystrata:tower` is pointed at `next-f6bc1acf`, which hasn't been done.

## Pass 22: a coding benchmark for Cyber-Tiel and Inkling 2 Mini (2026-11-24)

The aim is to compare the two models' coding without either one having an advantage it didn't earn. This is the
coding part only; thinking, pictures and long context come later. The task set is frozen: the prompts hash to
`315b1f5f46befe6b` (sha256 of every `tasks/*/task.json`, in order, the same on the Mac and on Tower), and the
graders and builders to `9c37d0ac92e41650` (`grade.py func_tasks.py run_bench.py repo_tasks.json build_repo.py`).
Nothing in those files is changed after seeing results. A broken task would be dropped for both models, and the
write-up would say so.

**Rules against cheating.**
- **No public benchmarks.** Every task is either a PolyStrata bug fixed after 2026-11-19, which is after both models
  were released, or a problem written for this pass.
- **Hidden tests decide.** The model never sees the tests. A repo fix passes when every test in the fix's test modules
  passes; a function passes when every hidden case is right.
- **Only the answer counts.** The reasoning text is not graded. A reply that never reaches an answer fails.
- **Same request for both models.** Each task's prompt is sent as the only user message, with thinking on and the same
  sampling for both: temperature 0.6, top_p 0.95, top_k 20, seed 1000 + sample number. Neither model card gives the
  server a default for Inkling, so neither model gets its card's sampling. The limit is 16,384 output tokens, of which
  at most 12,288 can be thinking, so an answer always has room.
- **Grading is separate.** It runs after both models, in the image with `--network none`.
- **Tokens and time are reported** beside the score.

**Repo fixes (13).** `build_repo.py` turns a fix commit into a task:
- the code at the commit's parent, with the bug;
- a bug report I wrote, in `repo_tasks.json`. It describes the symptom and what must not change, but not the fix. It
  names an interface only where the hidden tests call one the fix added (`runtime.recover_lost`, the `backend`
  argument, `ple_table_file`);
- excerpts of the files the fix changed: about 36,000 characters in all, placed at random around the changed lines
  and seeded by the task id, so the bug isn't always in the middle.

The model answers with SEARCH/REPLACE blocks, which have to match the file exactly once (trailing spaces ignored).

`grade.py check` turns the real fix into blocks the same way and checks two things on each task: the fix's tests fail
on the parent, and they pass with the fix. 18 commits were tried and 13 hold up the same way in two runs:
- r06, r17 and r18 were dropped: their test modules don't load in the image.
- r09 was dropped: its test checks the source text of the fix.
- r15 was dropped: its test needs the internals of the server's `Vision` class, which the prompt doesn't show.

Two tests fail with the real fix too, on this image (one each in r10 and r11), so they are allowed to fail. These are
the 13:

| Task | Commit | The bug |
|---|---|---|
| r01 | `05e60735e` | a stray backtick in the reasoning hides the tool calls after it |
| r02 | `a2497178b` | a quote right before `</think>` keeps the answer in the reasoning |
| r03 | `859f0da47` | a nested `<parameter>` pair cuts a tool argument short |
| r04 | `e626616e4` | an unexpected handler error drops the connection, not a 500 |
| r05 | `4f0e0f8b0` | a slot's live view restarts its clocks after a move |
| r07 | `cfc124573` | a stale runtime record makes `strata update` say a model is running |
| r08 | `840cbb012` | a setup run again loses `STRATA_RESIDENT_PIN=1` |
| r10 | `147cf024b` | a Hyper-V virtual adapter hides the NVIDIA card |
| r11 | `6b6098465` | an AMD setup is handed the CUDA package |
| r12 | `f6bc1acf1` | Inkling 2's prefill leaks `<|end_message|>` (Pass 21's fix) |
| r13 | `417d24af8` | Inkling 2's `strata_prefix` is a few tokens with thinking off |
| r14 | `f4afdd80d` | `--parallel` never reaches the engine |
| r16 | `12d491da4` | `make_room` drops the PLE shard's page cache |

**Functions (24).** `func_tasks.py` has 16 in Python and 8 in C++, each a spec with examples:
- **Python:** durations, SemVer order, a TTL cache, shell words, integer expressions (no `eval`), a token bucket, JSON
  Pointer, justified text, meeting rooms, build order, rotating rings, window medians (with a time limit), business
  days, npm-style version ranges, signed LEB128 and glob matching (no exponential backtracking).
- **C++:** inversions, a UTF-8 check, components, sliding maximum, checked 64-bit expressions, big-integer sums,
  top words and walls to break.

A seeded generator makes 80 to 500 hidden cases per problem. Every expected answer is computed by a reference
solution, never typed. Some cases are large (200,000 values, a 60 x 60 grid) to catch slow solutions.
`func_tasks.py selftest` checks that every reference passes its own cases, that an empty answer fails, and that the
examples in each prompt agree with the reference. All 24 pass.

**Scripts** (`p22/`, on Tower in `/mnt/user/appdata/polystrata/p22`):
- `model.sh MODEL LABEL SAMPLES [ids]`: starts the model in the test container with live's environment and runs
  `run_bench.py`. Cyber-Tiel uses `strata-cybertiel-next.json`, Inkling `strata-inkling2-mini.json`.
- `all.sh TAG SAMPLES [ids]`: both models, then grading, then live started again.
- `score.py`: the tables.

The image is `polystrata:next-f6bc1acf` for both models.

**Trial run (6 tasks, 1 sample each: r01 r12 p01 p05 c02 c05).** It checks the pipeline; the scores mean nothing
with this few. Every step worked:
- answers were kept;
- C++ compiled or its compile errors were reported;
- tests ran;
- a SEARCH block that matched nothing was caught;
- live came back.

Time for the 6 tasks: Cyber-Tiel 16 min (decode about 74 tok/s with sampling), Inkling 25 min (about 47 tok/s).
Hard tasks used the whole thinking budget on both models: 12,000 to 13,800 output tokens.

**What the trial found: the code quotes the models' own control tokens.** On r12, Inkling wrote its own
end-of-message token while thinking about the `<|end_message|>` bug. That ended its thinking after 363 characters;
the rest of its thinking went into the answer and ran to the 16,384-token limit with no SEARCH block.

The prompt itself is safe: `encode_prompt` keeps control-token text inside a message as plain text. What happened is
the model generating the real token. It is a fair weakness to count, since it happens to anyone who uses the model
on this code, but it doesn't hit both models equally:
- Cyber-Tiel's tags (`</think>`, `<tool_call>`, `<|im_start|>`) are in 7 prompts: r01-r03, r05 and r12-r14.
- Inkling's are in 2: r12 and r13.

So `score.py` also reports the 6 repo tasks with no control-token text (r04, r07, r08, r10, r11, r16), and the
per-task table marks a reply cut off at the limit as `L`.

**Full run, 1 sample of all 37 tasks (2026-11-24, tag `full1`).**
- Run times: Cyber-Tiel 14:25-15:43, Inkling 15:43-17:28. Grading took 2 min. Live was down from 14:25 to 17:30
  and came back loaded.
- Prompt hash `315b1f5f46befe6b` and script hash `9c37d0ac92e41650`, both unchanged since the trial.
- Graded rows, without the answers: `p22/results/full1-*.graded.jsonl`. The answers themselves are on Tower in
  `p22/runs/`.

| | Cyber-Tiel Coder 35B-A3B UD-Q4_K_XL | Inkling 2 Mini UD-Q4_K_XL |
|---|---|---|
| repo fixes | 2 of 13 | 4 of 13 |
| repo fixes with no control-token text | 1 of 6 | 3 of 6 |
| Python functions | 9 of 16 | 15 of 16 |
| C++ functions | 8 of 8 | 8 of 8 |
| all tasks | 19 of 37 | 27 of 37 |
| output tokens, median | 9,333 | 8,542 |
| answers of 12,288 output tokens or more | 16 | 14 |
| time per answer, median | 139 s | 176 s |
| generation time, all tasks | 1.29 h (67.9 tok/s) | 1.75 h (47.4 tok/s) |
| cut off at 16,384 tokens | 1 | 2 |
| repo answers whose blocks did not apply | 8 | 6 |

Per task (P pass, `.` fail, L cut off at the limit, A blocks did not apply):

| | c01-c08 | p01-p16 | r01 | r02 | r03 | r04 | r05 | r07 | r08 | r10 | r11 | r12 | r13 | r14 | r16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cyber-Tiel | `PPPPPPPP` | `.PP..PPPPPP...P.` | A | A | A | A | A | . | . | P | A | L | . | P | A |
| Inkling | `PPPPPPPP` | `PPPPPPPPPPPPP.PP` | A | P | A | P | A | L | A | P | P | L | A | . | . |

Every failure was read, to tell model mistakes from grader mistakes.

What went wrong, Cyber-Tiel:
- Python, 7 failures. Each one is the model's own bug:
  - p01 raised ValueError on extra spaces.
  - p04 returned `[]` for input with a tab.
  - p05 accepted `%( 5**+0)** -3`.
  - p12 hit an IndexError.
  - p13 wrote "`timedelta` needs importing" under its code but didn't add the import (NameError; 566 tokens).
  - p14 accepted a bad range.
  - p16 got a `*` glob wrong.
- Repo, 4 answers used the example file name from the format text. Three named `path/of/the/file.py` (r03, r11,
  r16) and one named `path/terminal/serve/server.py` (r04), so the grader could not find the file.
  - Graded again with the real file name put in, by hand and outside the scores above: r04 and r16 pass, r03 and
    r11 still fail. With that forgiven, Cyber-Tiel has 4 of 13 repo fixes and 3 of 6 neutral ones, the same as
    Inkling.
  - The scores above stay strict, because the format text asks for the file's path.
- Repo, broken format: r02 wrote SEARCH ... REPLACE ... REPLACE with no `=======` line.
- Repo, control-token tasks:
  - r02's thinking ended after 225 characters, at its own `</think>`.
  - r05 ended its answer with a `<tool_call>` to search the repo for a field name. No tools were offered.
  - r01's answer stops right after `=======`.
- r07, r08 and r13: the blocks applied, but the tests failed.
- r12 ran to the limit.

What went wrong, Inkling:
- Python: 1 failure. p14 accepted an invalid range.
- Repo, SEARCH did not match: r01 matched twice; r03 and r05 matched nothing.
- r08 (neutral): the whole block was written on one line with `\n` escapes, so there was no `=======` line.
- Repo, syntax errors after the blocks applied:
  - r07 left a `=======` line inside `updater.py`; the answer also ran to the limit.
  - r14 wrote `[*cfg.get("args") or [], ...]`, which is invalid Python.
- Repo, control-token tasks:
  - r12 wrote its own `<|end_message|>` while thinking and ran to the limit with no block, as in the trial.
  - r13 put `<|end_message|>` into the block's file name.
- r16: the blocks applied, but the tests failed.

**What this says, from one sample (each task's result is a single draw at temperature 0.6, so one task more or
less is noise):**
- Small self-contained functions: both models got every C++ task right.
  - In Python, Inkling made 1 mistake and Cyber-Tiel 7.
  - Cyber-Tiel's Python mistakes are edge cases it missed, not format problems.
  - That gap of 6 is the clearest result here.
- Bug fixes in real code: both are weak.
  - Inkling fixed 4 of 13 and Cyber-Tiel 2. If the file name it copied is forgiven, Cyber-Tiel also has 4.
  - Most repo failures are answers that could not be applied: wrong file names, SEARCH text that does not match,
    or a broken block. Fewer are wrong fixes.
- Cyber-Tiel thinks longer and decodes faster: a median of 30,000 thinking characters at 68 tok/s, against Inkling's
  18,000 at 47 tok/s. Over these 37 tasks it finished in 74% of Inkling's time.
- Writing their own control tokens hurt both models, but Cyber-Tiel more often. Its tags appear in 7 of these
  prompts and Inkling's in 2.

**Next, if wanted:**
- A second sample, to see which of these results hold.
- A format text whose example path cannot be copied as is. That changes the prompts, so its scores would not
  compare with `full1`.

**Not covered:** the agent loop (read, patch, run tests, try again), tool calls, CUDA code, Inkling without the
small-card settings, and greedy decoding.

### Pass 22, thinking: generated puzzles, three thinking settings (2026-11-24)

**The puzzles.** `p22/think.py` makes 36 puzzles from fixed seeds: 6 families, 3 levels each, 2 puzzles per level.
They are new, so no model can have seen them. Each has one exact answer, which a program computes.

| family | level 1 | level 2 | level 3 |
|---|---|---|---|
| logic grid (houses in a row) | 3 houses, 3 kinds | 4 houses, 4 kinds | 5 houses, 4 kinds |
| scheduling (tasks, prerequisites) | any number of workers, 6 tasks | 2 workers, 6 tasks | 3 workers, 8 tasks |
| shortest path (roads as text) | 6 towns | 10 towns | 14 towns, a closed road, a town to pass through |
| dates | days after a date, with weekday; days between | working days with holidays | a flight across time zones near a daylight-time change |
| word problems | rolls left; days to read a book | pipes and a drain (fraction); sugar mix (fraction) | shop profit (exact dollars); cyclists meeting (HH:MM:SS) |
| unit conversions | mpg to km/L; feet and inches to cm | mpg to L/100 km; psi to kPa | pump power in W; average speed in m/s |

- Every grid clue that the answer doesn't need is removed, and a grid whose answer is stated in a clue is rejected.
- A scheduling puzzle with workers is rejected if the simple lower bound already gives its answer.
- Daylight-time rules and unit definitions are written in the prompt, so the task is the arithmetic, not
  remembering rules.
- No prompt contains either model's control tokens.

**Checks.** `python3 think.py selftest` computes every answer a second way:
- grids: the solver confirms exactly one solution;
- scheduling: brute force over every start order;
- shortest path: Floyd-Warshall against Dijkstra;
- time zones: from the rule text in the prompt against the tz database;
- dates, words and units: a separate formula.

It also confirms the grader accepts each right answer and rejects a reply with no answer. All 36 pass, plus 9
grader cases (for example, `41 2/3` is not `125/3`, and `-$1,234.50` matches `-1234.5`).

**Asking and grading.**
- Each puzzle is asked three ways, so 108 requests per model:
  - `off`: thinking off (`chat_template_kwargs.enable_thinking: false`), up to 4,096 tokens;
  - `2k`: a 2,048-token thinking budget, up to 4,096 tokens;
  - `big`: a 12,288-token budget, up to 16,384 tokens.
- Sampling is the same as the coding set: temperature 0.6, top_p 0.95, top_k 20, seed 1000.
- Only the reply's last `ANSWER:` line counts. Numbers are compared exactly, as fractions. Rounded answers must
  round to the right digit, and dates and times must match exactly.
- Prompt hash: `cat think-tasks/*/task.json | sha256` = `3b7cc2b76bb3ce8f`.

**Scripts.**
- `run_bench.py` now takes a task's own `request` settings. Coding tasks have none, so their requests are as before.
- `model.sh` takes `TASKS=think-tasks`.
- `think.sh TAG SAMPLES [ids]` runs both models, then starts live again.
- Grading needs only python3: `think.py grade think-tasks runs/TAG-*.jsonl`, then `think.py score`.

**Trial run (4 level 2-3 puzzles × 3 settings, 1 sample each: grid3a, sched2a, date3a, word3b).** It checks the
pipeline; 12 answers per model mean little.
- Time: Cyber-Tiel 9 min (17:41-17:50), Inkling 9.5 min (17:50-18:00). Live was down 20 minutes.
- Thinking off worked on both models: no thinking text. A 2K budget ended the thinking where it should.

| | Cyber-Tiel off / 2k / big | Inkling off / 2k / big |
|---|---|---|
| grid3a (5 houses) | P P P | P . P |
| sched2a (2 workers) | L P P | P P P |
| date3a (Sydney to New York, the day after Sydney's change) | . P . | . P P |
| word3b (cyclists) | . P P | P P P |

- Cyber-Tiel wrote 21:00 on date3a with 12K and with thinking off, which misses Sydney's change to UTC+11 the day
  before. With 2K it got it right, which shows that one sample is noise.
- Inkling with 2K answered "Ben" on the grid. Its thinking ended at the budget, 11 characters before the answer.
- Cyber-Tiel with thinking off wrote its working into the answer and ran out of its 4,096 tokens on sched2a.

**Full run, 1 sample of all 36 puzzles × 3 settings (2026-11-24, tag `tfull1`).**
- Run times: Cyber-Tiel 18:07-18:54, Inkling 18:54-20:02. Live was down from 18:06.
- Prompt hash `3b7cc2b76bb3ce8f`, unchanged since the trial.
- Graded rows: `p22/results/tfull1-*.graded.jsonl`.

| puzzles right | Cyber-Tiel off | 2k | big | Inkling off | 2k | big |
|---|---|---|---|---|---|---|
| logic grid | 4 | 5 | 6 | 5 | 3 | 6 |
| scheduling | 3 | 5 | 5 | 3 | 4 | 6 |
| shortest path | 2 | 5 | 6 | 4 | 4 | 6 |
| dates and time zones | 0 | 5 | 5 | 3 | 6 | 6 |
| word problems | 5 | 6 | 6 | 3 | 6 | 6 |
| unit conversions | 4 | 6 | 5 | 5 | 5 | 6 |
| level 1 (of 12) | 7 | 12 | 12 | 10 | 12 | 12 |
| level 2 (of 12) | 6 | 11 | 12 | 8 | 10 | 12 |
| level 3 (of 12) | 5 | 9 | 9 | 5 | 6 | 12 |
| **all (of 36)** | **18** | **32** | **33** | **23** | **28** | **36** |
| output tokens, median | 337 | 1,989 | 2,315 | 671 | 2,073 | 2,200 |
| time per answer, median | 5 s | 24 s | 30 s | 16 s | 36 s | 37 s |
| time for all 36 | 5 min | 14 min | 27 min | 15 min | 19 min | 33 min |
| decode speed | 74 tok/s | 75 tok/s | 80 tok/s | 55 tok/s | 54 tok/s | 57 tok/s |
| cut off at the limit | 2 | 1 | 0 | 5 | 0 | 0 |

Every failure was read. They are the models' own errors, with one exception: word2b with thinking off (below).

**What went wrong, Cyber-Tiel:**
- Thinking off, 18 failures.
  - All 6 date puzzles were wrong. It got the weekday or the date wrong, or counted days off by one.
  - One flight got the right time but the wrong day. On the other it used standard time where daylight time had
    already started: it gave 14:05 for a Los Angeles departure on the morning of the change.
  - The rest were path, scheduling and grid answers it guessed from too little working, and `150.58` for a km/L
    value of 15.05, a tenfold slip.
- 2K budget, 4 failures:
  - date2b ran to the 4,096-token limit with no answer.
  - grid3a was wrong.
  - path3a gave 181 for 164.
  - sched3b gave 12, below the true minimum of 13; no schedule can do it in 12.
- 12K budget, 3 failures:
  - date3b made the same daylight-time slip as with thinking off.
  - sched3b gave 14 after 13,019 tokens.
  - unit3a gave 4350.8 for 4350.9, a rounding slip.

**What went wrong, Inkling:**
- Thinking off, 13 failures.
  - 5 of them ran to the limit: Inkling writes its working into the reply, and the grid and scheduling puzzles
    did not fit in 4,096 tokens.
  - Other errors:
    - two answers with the wrong weekday or day count;
    - a flight one day and one hour out;
    - `46 km` written after 7 tokens, with no working;
    - word3b's meeting time, also given after 10 tokens;
    - a profit of 4392.57 for 4395.77;
    - 15.06 for 15.05.
- 2K budget, 8 failures. Seven of them stopped at about 2,073 tokens, which is the thinking budget closing, with
  a guess for the answer (3 grids, 2 paths, 2 schedules); the eighth, 8.45 for 8.46, is a rounding slip.
- 12K budget: no failures.

**One fault in the prompt: word2b, the sugar mix.** It asks for "the exact percent (a fraction is fine)".
- With thinking off, both models answered with the right amount written as a share of 1, not a percent:
  Cyber-Tiel `9/65 ≈ 13.85%`, Inkling `9/65%`. The right answer is 180/13 %, which is about 13.85%.
- Both are counted wrong, as written. They are the only replies affected, one per model.
- A later version should ask for a decimal percent.

**What this says, from one sample:**
- With room to think, Inkling solved all 36 puzzles, including all 12 at level 3. Cyber-Tiel solved 33 and 9 of
  the 12 at level 3.
- Inkling needs that room. With a 2K budget its thinking was cut off on 7 hard puzzles and it guessed, so it fell
  to 28. Cyber-Tiel lost only 1 puzzle going from 12K to 2K: its median thinking is about 3,000 characters, against
  Inkling's 5,000.
- With thinking off, both drop a lot: Cyber-Tiel to 18 and Inkling to 23.
  - Cyber-Tiel answers in a few hundred tokens and makes careless errors; it got none of the 6 dates right.
  - Inkling works the problem out in its reply, then runs out of room on long ones.
- Cyber-Tiel is quicker for the same setting: 74-80 tok/s against 54-57, and the whole 108 took 47 min against
  67 min.
- The daylight-time rules were in the prompt, and still Cyber-Tiel used standard time on the Los Angeles flight
  both with thinking off and with 12K (it got it right with 2K).

### Pass 22, images: generated pictures, thinking off and a 2K budget (2026-11-24)

**The pictures.** `p22/img.py build OUT` draws 35 pictures from fixed seeds: 7 kinds, 5 of each, harder from the
first to the fifth.
- It runs in the PolyStrata image, because it needs Pillow, and uses Pillow's own font.
- Each picture has one exact answer, which the program knows.

| kind | 1 → 5 |
|---|---|
| charts | a bar's value; the second-highest bar; highest minus lowest (12 bars); the lowest month on a line; months where the blue line is above the orange (2 lines, 12 months, never closer than 8) |
| counting | "how many green triangles", 8 to 36 shapes in up to 5 colors and 5 shapes, none overlapping |
| reading | a code of 6, 8 and 10 characters, then one line out of 3 and out of 5; no look-alike characters (0/O, 1/I/L, 2/Z, 5/S, 6/G, 8/B) |
| tables | a cell (4×3, 6×4); the largest in a column (8×5); a row's sum (10×6); values over 500 in a column (12×6) |
| code with one bug | which line of a Python function is wrong (7 to 16 lines with line numbers); each bug is checked by running it: the right function passes its tests, the shown one fails |
| a settings window | the greyed-out button (4, 6); ticked boxes (6); switches that are on (8); a text field's value |
| grid of shapes | what is left of, above or below a shape (3×3, 5×5); a shape's row and column (4×4); how many other shapes share its color |

**Asking and grading.**
- Each picture and question go in one OpenAI `image_url` message: 70 requests per model.
- `off`: thinking off, up to 4,096 tokens. `2k`: a 2,048-token thinking budget, up to 4,096 tokens.
- Sampling and grading are the same as for the puzzles: `think.py grade`, with answer kinds for codes, label
  lists, "color shape" and "row, column".
- Prompt hash: `cat img-tasks/*/task.json | sha256` = `d5cdf3fabeef904e`.

**What the image limit means for each model.** Both configs have the same `"vision": {"max_tokens": 300}`, which
the image reader gets as `--max-tokens`. Each model's own reader decides how many tokens a picture takes.
- Measured with the same questions sent without their picture (`PROBE=` in `model.sh`):
  - the chart question is 229 tokens for Cyber-Tiel and 54 for Inkling;
  - the grid question is 242 and 66.
- So a picture takes:

| picture | Cyber-Tiel | Inkling |
|---|---|---|
| 800×560 (charts, counting, reading, the window) | about 280 | about 280 |
| 540×540 (grid of shapes) | about 290 | about 200 |
| 800×260 (smallest table) | about 200 | about 140 |

- Inkling's reader doesn't reach 300 on any of these pictures, so the limit doesn't bind it. Raising it to 1,024
  left every Inkling prompt the same size (below).
- An earlier version of this note said Inkling sees every picture in less detail. That was wrong: only the square
  and short pictures get fewer tokens.

**Trial run (5 pictures × 2 settings: chart5, read5, ui3, space3, bug5).**
- It ran right after the thinking run: Cyber-Tiel 20:03-20:05, Inkling 20:05-20:08. Live came back at 20:08.
- Every reply had an `ANSWER:` line, and the grader read all 20 correctly.

| | Cyber-Tiel off / 2k | Inkling off / 2k |
|---|---|---|
| chart5 (two lines) | P P | . P |
| read5 (line C of 5) | P P | P P |
| ui3 (ticked boxes) | P P | P P |
| space3 (row and column) | P P | P P |
| bug5 (16-line merge) | P P | . . |

- Inkling with thinking off answered in 6 to 16 tokens; it gave 7 for 8 on the chart and line 7 for line 15.
- With 2K, Inkling's thinking ran to the budget on bug5 and it answered line 14. Cyber-Tiel found line 15 in 283
  tokens.

**All 35 pictures, and three Inkling settings (tag `ivar1`, 2026-11-24 20:29-21:02).** `p22/inkvar.sh` ran four
rounds, each with 1 sample of all 35 pictures × 2 settings:
1. Cyber-Tiel with its config.
2. Inkling with setup's config.
3. Inkling with a picture limit of 1,024 (`configs/p22-ink-v1024.json`).
4. Inkling with that limit and without the three small-card settings (`configs/p22-ink-acc1024.json`, which is
   Pass 21's `acc.json` with the limit raised).

The test configs are copies in `/data/config/p22-*`. Neither model's config was changed.

| pictures right, of 35 | Cyber-Tiel | Inkling, setup's | Inkling, limit 1,024 | Inkling, 1,024, no small-card settings |
|---|---|---|---|---|
| thinking off | 31 | 27 | 27 | 27 |
| 2K budget | 33 | **34** | **34** | **34** |
| output tokens, median (off / 2k) | 30 / 123 | 6 / 163 | 6 / 166 | 6 / 190 |
| time for all 70 | 5.8 min | 7.7 min | 7.0 min | 8.8 min |

| by kind, right of 5: off / 2k | Cyber-Tiel | Inkling (all three settings) |
|---|---|---|
| charts | 5 / 5 | 4 / 5 |
| counting | 3 / 5 | 2 / 5 |
| reading | 4 / 4 | 5 / 5 |
| tables | 5 / 5 | 3 / 5 |
| code with a bug | 4 / 4 | 3 / 4 |
| settings window | 5 / 5 | 5 / 5 |
| grid of shapes | 5 / 5 | 5 / 5 |

**Every failure, read:**
- Both models, both settings: `bug3` (the vowel counter), which both answered as line 9 for line 8.
- Cyber-Tiel: read3 copied `HRWYYMMHVC` as `HRWYYMHVC` (one M short) in both settings. With thinking off it also
  missed counting 2 and 3, by one each.
- Inkling with thinking off, the same 7 misses in all three settings:
  - charts: 7 months for 8;
  - counting: three counts one too high or low;
  - tables: a row sum of 2,444 (2,697 without the small-card settings) for 3,247, and 11 values over 500 for 8;
  - the merge bug: line 9, 14 or 10 for 15.
  - It answered each in about 6 tokens, with no working.

**What this says:**
- With thinking on, Inkling reads these pictures at least as well as Cyber-Tiel: 34 against 33. Inkling thinks on
  a plain request with no thinking setting, which is the default, so this is what it does out of the box.
- With thinking off, Inkling falls to 27 and Cyber-Tiel to 31. Inkling's misses there are counting and arithmetic
  done with no working. It gets the same pictures right when it thinks, so seeing them is not the problem.
- Neither setting helped:
  - A picture limit of 1,024 changed nothing: Inkling's reader never reached 300 here.
  - Turning the small-card settings off changed no score and was 14% slower.
- So the best setting for Inkling on images is the one it already has, with thinking on.
- Not tried: `"min_tokens"`, which scales a picture up so it takes more tokens. These failures aren't about
  detail, so it would need harder pictures to show anything.
- The trial's 7 of 10 for Inkling came from 2 of these 5 pictures with thinking off and one 2K run that hit its
  budget. Over all 35 pictures that run (bug5, 2K) was right.

### Pass 22, long context and stability (2026-11-24)

**The tasks.** `p22/long.py` builds them on the Mac from the repository.
- **Long context.** The document is PolyStrata's own Python: `terminal/serve` and `terminal/strata_control` at
  `f6bc1acf1`, the commit the test image was built from, with tests left out.
  - Files are picked at random until the document reaches a size: about 50K, 100K or 150K characters. A fourth
    size, about 400K, is for Cyber-Tiel only, because Inkling's config stops at 65,536 tokens.
  - Notes are put in as comments at set depths:
    - 18 lines saying which depot a unit is kept at;
    - 6 saying which town each depot is in;
    - 4 to 6 changes to each of two queues' retry limits.
  - Six questions follow the code. Each is asked on its own, with the code first so the engine can reuse it:
    - q1: one note 10-30% in;
    - q2: one note 70-95% in;
    - q3: two notes, the town of a unit's depot;
    - q4: how many units are at one depot (3 to 6 notes);
    - q5: the last of one queue's changes, with the other queue's changes in between;
    - q6: the value of a real constant in the code, the only assignment of that name.
- **JSON.** 10 short texts about a person, written as a JSON object of 5 fields:
  - `free`: with only the instruction, "Reply with only the JSON object";
  - `schema`: with a `response_format` JSON schema, which the server's grammar enforces.
  - The texts have traps: a town the person moved away from, and a brother who lives elsewhere.
- **Settings.** All tasks use a 2,048-token thinking budget, up to 4,096 tokens, with the same sampling as the
  rest of Pass 22.
- **Checks.** `python3 long.py selftest` finds every answer again in the finished document. Notes are read back
  with a pattern, and the constant with Python's `ast`, file by file. All 4 sizes pass.
  - The JSON grader takes the whole reply, with or without a code fence, and needs every field equal and of the
    right type.
- Prompt hash `5484d97ec0927b99`.

**Run (tag `lfull1`).** Cyber-Tiel ran 21:17-21:27 and Inkling 21:27-21:42; live came back at 21:42.

| | Cyber-Tiel | Inkling |
|---|---|---|
| prompt, 50K / 100K / 150K characters | 14,781 / 26,039 / 39,044 tokens | 17,377 / 32,560 / 49,002 tokens |
| right at 50K (of 6) | 5 | 5 |
| right at 100K | 6 | 6 |
| right at 150K | 6 | 6 |
| right at 400K (99,933 tokens) | 4 | not asked: turned away with 400, as expected |
| first question at each size, s (reads the code) | 10 / 16 / 20 / 52 | 38 / 96 / 73 / - |
| later questions, median s | 8 | 26 |
| JSON, `free` / `schema` (of 10) | 10 / 10 | 10 / 10 |
| `free` replies that are the bare object (no code fence) | 7 | 10 |

The misses, all read:
- At 50K, both miscounted the units at Teller (q4): Cyber-Tiel said 5 and Inkling 7, for 6.
- At 400K, Cyber-Tiel:
  - counted 3 units for 4 (q4);
  - gave the other queue's last change, 39, for 9 (q5).
- Every note question at 100K and 150K was right on both, and every constant.
- Inkling's thinking ran to the 2,048 budget on 7 of these questions, mostly q4 and q5, and still answered all
  but one of them right.

What this says, from one sample:
- Up to 49,000 tokens, both find single notes, chains of two and constants anywhere in the code. Counting notes
  spread through the code (q4) is the weak spot for both.
- Inkling's tokenizer makes the same code about 18-25% more tokens: 49,002 against 39,044 at 150K characters.
  Its 65,536-token limit therefore holds about 200K characters of this code.
- Inkling reads long prompts about 3 times slower, as in Pass 21.
- Cyber-Tiel still answers most questions right at 100,000 tokens, but made its two worst mistakes there.
- With JSON, both got all 20 right with and without the schema. Cyber-Tiel put 3 of its 10 `free` answers in a
  code fence, which a strict parser would refuse.

**Stability, every reply in Pass 22.** `python3 long.py loops RUNS...` counts:
- loops: the same line of 20 or more characters 4 times in a row, or an ending that repeats 4 times;
- replies cut off at the token limit;
- failed requests.

| | coding (37) | puzzles (108) | pictures (70) | long + JSON (44) |
|---|---|---|---|---|
| Cyber-Tiel: loops / cut off / failed | 0 / 1 / 0 | 0 / 3 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Inkling: loops / cut off / failed | 0 / 2 / 0 | 0 / 5 / 0 | 0 / 0 / 0 | 0 / 0 / 6 (the 400K prompts, turned away by design) |

- I read the last part of each reply that hit the limit. All were still working, not stuck: some were mid-proof,
  some still searching a schedule.
- Cyber-Tiel's cut-off coding reply ended "Let me read", reaching for a tool it wasn't given.
- Across Pass 22 each model answered every request it was sent with no error: 259 per model in the four full
  runs, plus the trials and Inkling's two extra image rounds. Cyber-Tiel loaded 7 times and Inkling 9.

## Pass 23: more context for Inkling 2 Mini (2026-11-25)

**Why 64K.** Inkling's config stops at 65,536 tokens because of the card, not RAM:
- 8 of its 48 layer runs look at the whole context and keep 32,768 bytes a token on the card (17,408 with
  `--kv q8_0`). At 128K that is 4 GiB (2.1 GiB in Q8_0), at 1M 32 GiB.
- The engine can keep a cache in host RAM (`--kv-resident`) only for qwen35moe, so the RAM the newer engine freed
  does not help Inkling.
- The cache grows on the card as a conversation grows, taking VRAM from the expert cache. So a config of 256K or
  512K loads with the same experts on the card as 64K (checked: 1,126 and 1,124 of 3,072), and runs out of room only
  when a conversation gets long.

**What fixes it.** Open upstream PR #336 (`STRATA_INK_ZHA`, opt-in):
- It moves those 8 runs' keys and values to pinned RAM as 4-bit leaves, 9,216 bytes a token.
- The card keeps a summary of every 64 keys (592 bytes a token) and the last 1,024 keys.
- A token reads exactly:
  - chunk 0;
  - its last 1,024 keys;
  - the 16% of far chunks whose summaries score best.
  The rest it reads through the summaries, so the answers move a little.
- It merged cleanly onto `tower-next` (f6bc1acf1): branch `tower-ctx`, image `polystrata:ctx-a583121c`, built in 6
  minutes with the same Dockerfile.

The other open PRs about Inkling were read for this question:
- #300, #224, #299 and #271 each measured 0.5-3% or worse on long prompts.
- #320 (the prompt batch picked from free VRAM) conflicts with #336, because both size the prompt buffers and #336
  sizes its ring from a fixed `--prompt-batch`. A fixed batch was tested instead (below).

**Setup.**
- Configs in `/data/config`, copies of `strata-inkling2-mini.json` (live's, untouched) with `--max-context` changed:
  - `p23-ink-zha-1m.json`: the config's `"env"` sets `STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin`;
  - `p23-ink-zha-1m-pb2048.json` and `-pb4096.json`: the same with `--prompt-batch`;
  - `p23-ink-zha-512k-pb2048.json`;
  - `p23-ink-q8-128k.json`: no index, `--kv q8_0`, the control.
- Tasks: `p23/ctx.py`, Pass 22's long-context questions (`p22/long.py`) at four more sizes. Files that hold a
  planted name (Swift, Wren, ...) are left out at these sizes. The sizes are 400K, 750K, 1.5M and 3M characters of
  PolyStrata's Python: 127,956, 249,741, 476,742 and 934,351 Inkling tokens.
  - q1, q2: one note early or late;
  - q3: two notes;
  - q4: a count of notes;
  - q5: the last of a queue's changes;
  - q6: a constant in the code.
  Thinking on, 2,048-token budget, one sample.
- Runs: `p23/run.sh LABEL CONFIG [IDS]` (through `p22/model.sh`, port 5003, live stopped and started again after)
  and `p23/chain*.sh`.

**The ladder: index on, 1M context, default prompt batch (1,024).** 1,028 of 3,072 experts on the card at load (4.3
GiB), against 1,126 without the index.

| Level | Prompt | Read | First question | Written after | Right |
|---|---|---|---|---|---|
| 5 | 127,956 tokens | 828 tok/s | 2.6 min | 43.8 tok/s | 5 of 6 |
| 6 | 249,741 | 800 | 5.2 min | 39.4 | 4 |
| 7 | 476,742 | 763 | 10.4 min | 39.2 | 5 |
| 8 | 934,351 | 702 | 22.2 min | 38.5 | 3 |

- Every later question reused the prompt (126,976 to 933,888 tokens) and answered in 8-62 s.
- The misses:
  - q4 (the count) at 250K, 477K and 934K;
  - q5 (the last change) at 128K, 250K and 934K;
  - q3 (two notes) at 934K.
  Pass 22 found q4 the weak spot for both models already at 50K, and Cyber-Tiel missed q4 and q5 at 100K.
- VRAM peaked at 7,741 MiB. Pinned RAM: loading the 1M config took 9.4 GB more than live's (the leaves are pinned
  for the whole context at load). At least 16.9 GB of RAM stayed available through the 934K level.

**The control: no index, `--kv q8_0`, 128K.** Same image, same 128K level:
- read at 695 tok/s and wrote 30.0-31.2 tok/s;
- right on 4 of 6, missing q4 (said 4 for 5) and q5 (20 for 3).

So at 128K the index read 19% faster and wrote 45% faster than full attention. It got as many questions right or
more: 5 against 4, one sample.

**Prompt batch.** 128K level, index on:

| `--prompt-batch` | Experts on the card at load | Read | Written after | q1, q2 |
|---|---|---|---|---|
| 1,024 (default) | 1,028 | 828 tok/s | 43.8 tok/s | right |
| 2,048 | 852 | 986 | 44.1 | right |
| 4,096 | 488 | 465 | 43.0 | right |

**Short prompts.** Three prompts of 36-50 tokens, thinking off, up to 512 tokens written, on the fixed image. The
first request after a load is slow in every config.

| Config | 1st: read, written | 2nd | 3rd |
|---|---|---|---|
| live's (64K, no index) | 15.6 s, 42.5 tok/s | 0.9 s, 50.3 tok/s | 470 tokens in 10.2 s (log cut short) |
| index, 1M, batch 2,048 | 18.9 s, 32.3 | 1.8 s, 41.6 | 2.0 s, 42.5 |
| index, 512K, batch 2,048 | 18.6 s, 32.4 | 1.8 s, 41.9 | 1.8 s, 43.0 |
| index, 1M, batch 1,024 | 19.8 s, 28.7 | 1.9 s, 42.1 | 2.3 s, 41.6 |

With the index a short chat writes about 15% slower (42 against 46-50 tok/s), and a short prompt takes about a
second longer to read. The read maps the index's prompt buffers at their full size, sized for `--max-context`, and
takes the VRAM from the expert cache. After the read it gives the VRAM back.

**A bug in #336, fixed on `tower-ctx`.**
- **The bug.** With the index on, a parked conversation kept none of its full layers' rows (`card_row_bytes()` was 0
  for them). The leaves are stored by position, so the next conversation writes over them. Bringing the first
  conversation back rebuilt its ring from the other conversation's leaves, and kept the other conversation's chunk
  summaries.
- **The test.** `p23/ctx.py swap`: two different ~128K-token texts, asked A, B, A, A, B. On image `ctx-a583121c`
  (#336 as it is):
  - A's two later questions were wrong. The model quoted notes that are only in text B ("unit LINNET-52 is kept at
    depot Mirel"), said A's unit "is not stated in the provided files", and named the wrong town.
  - The engine reported no error.
- **The fix.** Commit `e20f5264` on `tower-ctx`:
  - a parked conversation keeps the leaves (9,216 bytes a token, less than the 32,768 the card's rows take without
    the index);
  - bringing it back writes the leaves back and rebuilds the summaries from the first row it loads.
  - A first version (13bf34f6) read every slot's leaves from the first slot's place. It made the answers nonsense
    (the model looked for an image in the prompt) and was replaced.
- **With the fix.** All 5 answers were right.
  - A 128K conversation parks in 0.9-1.3 s as 2.2 GB (1.0 GB before).
  - Coming back to it costs 11-16 s for the summaries' rebuild, against 3.5 s when nothing is rebuilt.

**The ladder again: the fixed image (`ctx-e20f5264`), 1M, `--prompt-batch 2048`.** 850 experts on the card at load.

| Level | Read | First question | Written after | Right |
|---|---|---|---|---|
| 5 (127,956 tokens) | 993 tok/s | 2.1 min | 44.0 tok/s | 5 of 6 |
| 6 (249,741) | 782 | 5.3 min | 41.9 | 4 |
| 7 (476,742) | 744 | 10.7 min | 38.6 | 5 |
| 8 (934,351) | 676 | 23.0 min | 33.4 | 3 |

- The same questions were missed as in the first ladder, except at 128K: this time q4 was wrong (said 2 for 5) and
  q5 right.
- Batch 2,048 reads faster only at 128K (993 against 828 tok/s). At 250K and longer it reads 2-4% slower than
  1,024, and it keeps 178 fewer experts on the card.
- After the first read at 128K, the first follow-up question re-read 21,459 tokens (26 s). With 1,024 the follow-up
  reused 126,976 tokens, so that difference is probably the checkpoint spacing.
- At 477K (level 7), in both ladders, q1's reply had part of the thinking in the answer, with a raw `<|end_message|>`
  between them. The final `ANSWER:` line was right both times. That document holds PolyStrata's own Inkling reply
  parser, whose code has the template's tags in its strings. The model quoted that code while it thought, and the
  server's split between thinking and answer went wrong. This comes from the test text, not from the index.

**What this says.**
- With #336 and the fix, Inkling 2 Mini reads and answers about 934,000 tokens on the RTX 3070. Context costs about
  9 KB of pinned RAM a token, not VRAM.
- It finds single notes and constants at every size. Two-note chains hold up to 477K. Counting notes and the last of
  several changes were already weak at 50K and stay weak.
- Up to 477K it gets as many right as at 128K. At 934K it gets 3 of 6.
- On long prompts the index is faster than full attention:
  - at 128K, 828 against 695 tok/s read and 44 against 30 tok/s written;
  - written speed stays at 38-44 tok/s up to 934K tokens.
- What it costs:
  - about 15% of the writing speed on short chats;
  - a second more on a short prompt's read;
  - pinned RAM for the whole `--max-context` at load: 9.4 GB measured at 1M; at 512K it should be about 4.5 GiB (9,216 bytes × 524,288, not measured);
  - 11-16 s when an older parked conversation comes back;
  - #336 itself is an open PR, and had the bug above.
- The default prompt batch (1,024) is the better setting.

**Not done.**
- Nothing changed on live: config, template and image are untouched, and `polystrata:tower` still runs Cyber-Tiel.
- `tower-ctx` (`tower-next` + #336 + the fix) is not merged into `tower-next`.
- Nothing was posted upstream. The bug is #336's own, so the PR's thread is where to report it.
- Not measured:
  - more than one sample a question;
  - a real agent conversation with tool calls;
  - images with the index on;
  - Cyber-Tiel at these sizes.

## Pass 24: one Inkling config for short chats and long documents (2026-11-25)

**Question.** Can one ZHA config at 1M tokens also serve short chats, so Inkling needs no second config and no
engine reload to switch between them? Pass 23 measured short chats about 15% slower under the index (42 against
50 tok/s through the server). This pass finds where that time goes and removes the part that was not needed.

**Setup.** Tower as in Pass 23. Images `polystrata:ctx-e20f5264` (Pass 23's) and `polystrata:ctx-97665a39`
(`tower-ctx` 97665a39: the change below). Scripts in `p24/`:
- `bench.sh`: strata-poly by itself on a 21-token prompt, 400 or 800 tokens at temperature 0;
- `chain1.sh`-`chain5.sh`: the runs below;
- `probe.sh`: the server with three short chats at temperature 0, asking the engine for its expert cache after each
  (`POST /v1/vram {}`).

**1. The engine by itself, short prompt** (one run each; `CPU experts` is the engine's own ms a token):

| Run | Experts on the card | Writes (tok/s) | CPU experts |
|---|---|---|---|
| plain 64K, after Cyber-Tiel ran | 1,120 | 58.7 / 57.6 | 6.3 / 6.1 ms |
| ZHA 64K | 1,031 | 52.8 | 6.8 ms |
| ZHA 1M, all RAM pinned at load | 1,025 | 49.6 / 47.6 | 10.9 / 11.4 ms |
| plain 64K, warm file cache | 1,120 | 61.5 | 3.7 ms |
| ZHA 512K, warm | 1,028 | 56.5 | 4.5 ms |
| ZHA 1M, warm | 1,025 | 55.2 | 4.9 ms |

- With the model file in the file cache, the index costs 1.4-1.8 ms a token (61.5 against 55-56.5 tok/s).
- Right after Cyber-Tiel, the 1M config's CPU experts took twice as long. Pinning 9.4 GB of leaves at load left the
  17.6 GB model file too little file cache; the 64K configs did not have that problem.
- `STRATA_INK_ZHA_POOL=0`, which #336 allows, stopped the engine with `an illegal memory access was encountered`.
  Not looked into.

**2. The change** (`tower-ctx` 97665a39, `inkling2.cpp`). The leaves' RAM is reserved address space at load (mmap)
and pinned 64K positions at a time as a read or an answer reaches them. A conversation under a quarter of what is
pinned gives the rest back (unpinned, `MADV_DONTNEED`). The conversation cache's automatic size still leaves the
whole context's leaves out, so it is the same 6,400 MiB as before. `STRATA_INK_ZHA_PIN_ALL=1` pins everything at
load, as before.
- The 1M engine loads in 7-20 s instead of 17-30 s.
- RAM used at rest with a short chat: about the 64K config's, not 9.4 GB more.

**3. Through the server, short chats** (`probe.sh`, image ctx-97665a39, the same three prompts at temperature 0):

| Config | Experts on the card after a request | Writes (tok/s) | Short follow-up read |
|---|---|---|---|
| plain 64K | 1,246 | 55.0 / 55.8 | 0.31-0.35 s |
| ZHA 1M | 1,157 | 49.6 / 49.5 | 0.35-0.43 s |

- The index costs 2.1 ms a token here, 11%.
- 89 fewer experts are on the card: the index's chunk cache (600 chunks for each of the 8 full layers, about 350 MB)
  and its ring. Part 5 shows that this is only a small part of the 2.1 ms.
- Pass 23's short-chat runs (`short-tasks`, through `run_bench.py`) gave 42-43 against 49-51 tok/s, and the same
  tasks on the new image gave the same: 42.1-42.6 with RAM pinned as it grows, 41.5-44.3 with all of it pinned at
  load. The page-cache effect of part 1 did not show up through the server in those runs.

**4. Long conversations still work.** On image ctx-97665a39, 1M config:
- 128K question, two short chats, a second 128K question: both answers right (Varn, Mirel). The 128K
  conversation was parked (2.2 GB) and restored in 0.16 s.
- 477K question (read at 761 tok/s, written at 39.5 tok/s), a short chat, a second 477K question: both right (Quorn,
  Mirel). The short chat gave the long conversation's pinned RAM back: its 49-token read took 5.1 s instead of
  1-2 s, and RAM in use peaked at 12.8 GB against about 15 GB with the leaves kept (an estimate: 4.8 GB of leaves
  plus the 5.4 GB parked copy). The parked 475,136 tokens came back in 2.75 s and pinned again.

**5. The chunk cache grows with the context** (`tower-ctx` 513eceb1, image `polystrata:ctx-513eceb1`). The cache
starts at 64 chunks a KV head and doubles as the conversation's chunks pass it, up to `STRATA_INK_ZHA_POOL` (600). It
takes the VRAM from the expert cache as the summaries do, and a much shorter conversation shrinks it and gives the
VRAM back. `zha_attend()` now reads through the cache when a selection over the chunks built so far fits, not one
over the whole `--max-context`.

| Config (`probe.sh`, image ctx-513eceb1) | Experts on the card after a request | Writes (tok/s) | Short follow-up read |
|---|---|---|---|
| plain 64K | 1,244 | 55.2 / 56.0 | 0.31-0.35 s |
| ZHA 1M | 1,234 | 49.9 / 50.7 | 0.36-0.41 s |

- 77 of the 89 experts came back, but writing went up by less than 1 tok/s. The gap left, about 1.9 ms a token
  (9.5%), is the index's own kernels: about 15 launches for each of the 8 full layers, each token.
- 128K question, two short chats, a second 128K question: both answers right (Varn, Mirel). The 128K read went at
  830 tok/s and its answer was written at 46.0 tok/s. The cache grew to 600 for it, shrank for the short chats and
  grew again when the 128K conversation was restored (0.16 s).

**6. A smaller ceiling** (`chain7.sh`, image ctx-513eceb1, `probe.sh`, the same three prompts):

| Config | Experts on the card | Writes (tok/s) | Conversation cache (auto) |
|---|---|---|---|
| plain 64K | 1,244 | 55.2 / 56.0 | 8,192 MiB |
| ZHA 256K | 1,236 | 50.6 / 51.5 | 8,192 MiB |
| ZHA 512K | 1,236 | 51.2 / 52.1 | 8,192 MiB |
| ZHA 1M | 1,234 | 49.9 / 50.7 | 6,400-6,656 MiB |

- 256K and 512K write about 1 tok/s faster than 1M on short chats (some of the index's per-token work is sized by
  `--max-context`); between them the difference is within one run's spread.
- Under 1M the conversation cache keeps its full 8 GiB: the leaves still to pin (4.5 GiB at 512K) leave enough.
- At 512K the gap to plain 64K is about 7% (51.7 against 55.6 tok/s).

**7. Before going live at 512K** (image ctx-513eceb1 unless named; `chain8.sh`-`chain11.sh`, `probe2.sh`):
- Inkling 512K, a 128K question, two short chats, a second 128K question: both right (Varn, Mirel); read at 834 tok/s,
  written at 46.0 tok/s, the parked 128K conversation restored in 0.17 s.
- Fresh prompts (nothing reused, three rounds; the first round is the engine's first request), rounds 2 and 3:

  | Prompt | Inkling 64K | Inkling ZHA 512K |
  |---|---|---|
  | 102 tokens | 1.6-1.8 s | 1.3-1.5 s |
  | 685 tokens | 1.9-2.1 s | 1.9-2.0 s |
  | 4,512 tokens | 6.8 s | 6.8-7.3 s |
  | 17,967 tokens | 18.9-19.1 s | 18.9-19.0 s |

- Pass 22's 35 pictures with thinking off, the same evening:

  | Image, config | Right | Median s an answer |
  |---|---|---|
  | live `polystrata:tower` (9172f75e), Inkling 64K | 28 of 35 | 5.8 |
  | ctx-513eceb1, Inkling 64K | 27 of 35 | 5.8 |
  | ctx-513eceb1, Inkling ZHA 512K | 28 of 35 | 5.9 |

  Pass 22's run of the same pictures on the live config took 2.4 s an answer; today's live image took 5.8 s too, so
  that difference is not the image or the index (not looked into).
- Cyber-Tiel's live config (`strata-cybertiel-next.json`), 105K-token question and two short chats: right on both
  images; the 105K read went at 1,940 tok/s on the live image and 2,021 on ctx-513eceb1. #336 changes
  `read_groups()` for every model; this is the one Cyber-Tiel check.

**8. Live since 2026-11-25 20:55** (the user's choice: one Inkling config at 512K).
- `polystrata:tower` = `tower-513eceb1` = `ctx-513eceb1` with the old live image's start command (Cyber-Tiel's
  config; `p24/Dockerfile.live-513eceb1`). The first swap used `ctx-513eceb1` itself, whose command starts Inkling;
  it was replaced within minutes.
- `strata-inkling2-mini.json`: `--max-context 524288` and `STRATA_INK_ZHA`; the 64K one is
  `strata-inkling2-mini.json.pre-zha`.
- `strata-cybertiel-next.json` touched (contents unchanged): three configs carry Cyber-Tiel's model name and the
  server lists the newest file, which was the 0.1.45-era `strata-cybertiel-ud-q4_k_xl.json` with an engine path the
  image does not have. Switching from Inkling back to Cyber-Tiel failed with it; now Cyber-Tiel -> Inkling -> Cyber-Tiel
  works on live.
- Branch `tower-next` fast-forwarded to `tower-ctx` 513eceb1.
- Rollback: `docker tag polystrata:tower-9172f75e polystrata:tower`, `rebuild_container PolyStrata`, and
  `strata-inkling2-mini.json.pre-zha` back to `strata-inkling2-mini.json`.

**What this says.**
- Through the server, a single Inkling config at 1M cost 11% of the writing speed on short chats (49.5 against
  55.5 tok/s) before part 5 and about 0.05 s more on a short read. It no longer holds 9.4 GB of RAM when the chats are short.
- Switching between two configs costs an engine reload (20-30 s), and a long conversation started on one is read
  again on the other. One 1M config is the simpler setup if the 11% is acceptable.
- With the chunk cache growing (part 5) the cost is 9.5%: 50.3 against 55.6 tok/s. What is left is the index's
  per-token kernels; fewer launches for short contexts would be the next thing to try.

**Not done.**
- Live changed only in part 8.
- The 477K run was not repeated on image ctx-513eceb1.
- One run a configuration; no answers graded on short chats.
- Nothing posted upstream. The #336 author pushed the parking fix independently (96a1a1180, 2026-11-25); the
  `STRATA_INK_ZHA_POOL=0` crash and this change are not reported.

## Shared-load changes (2026-10-17)

With Plex, Deluge and Duplicati running again, the GGUF (22.7 GB) has to stay cached next to them, and about 22.5 GB
is available. Each container's own file cache counts against its cgroup v2 memory limit, so a cap makes it drop its
own cached files rather than the model's.

- Plex `--memory=4g`, Deluge `--memory=1g`: added to ExtraParams in the dockerMan templates (backups:
  `*.xml.pre-memcap`) and applied live with `docker update`.
- The Docker folder is now `/mnt/cache/system/docker/` instead of `/mnt/user/...`, which skips shfs (backup:
  `/boot/config/docker.cfg.pre-cachepath`). Restarting Docker autostarts PhotoPrism and does not start
  `strata-cybertiel`, so fix both by hand afterwards.
- Plex: periodic library scans are off (folder watching stays on), done by the user. Transcoding stays on the CPU.

## Swap on the cache SSD (2026-10-18)

Tower had no swap. The model's pages never go to swap (they are a file's: the kernel drops them and reads them
again), but about 4.5 GB of program memory sits idle (`Inactive(anon)`, much of it Unraid's in-RAM root file
system). With swap the kernel can move that out and give the RAM to the page cache. The model's 21.2 GB sits in about
25 GB of available RAM, so the margin was 3-4 GB, and Pass 7 saw model pages read again with Plex and the add-on
generator running.

- `/mnt/cache/system/swapfile`, 8 GiB (`dd`, `mkswap`), `vm.swappiness=10` (only memory that sits idle goes out).
- It goes on when the disks are mounted and off before they unmount: an active swapfile on `/mnt/cache` would keep the
  cache from unmounting and the array from stopping. Hooks: `/usr/local/emhttp/webGui/event/disks_mounted/swapfile`
  and `.../unmounting_disks/swapfile`. They are in RAM, so `/boot/config/go` writes them at boot, before `emhttp`
  starts (backup: `/boot/config/go.pre-swap`). Both hooks were run by hand once: off, then on again.
- Not yet measured: whether the model's pages stay cached better with Plex and the add-on generator running.

The photo backup (User Scripts `photo_backup`, an `rsync` through `/mnt/user`) evicts the model whatever the swap: on
2026-10-18 only 9 of 21.2 GiB were left cached. Reading the file three times (`cat ... > /dev/null`, about 2
minutes) brought it to 20.6 GiB. The page cache keeps a page read once on its inactive list, so one pass is not
enough. That could go at the end of the backup script.

## Cleanup (2026-10-18)

- `strata-old` is gone (it was no longer on Tower).
- The disk2 copies of `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` and `mmproj-F16.gguf` are byte for byte the cache copies
  (`cmp`). They are kept on purpose.
- `strata-cybertiel` was recreated with `CONTEXT=262144 KV=int8` (see Pass 7). `strata` (Qwen3.6, 64K fp16) is
  stopped and kept. `strata-cybertiel` and `strata` both map port 8080: stop one before starting the other.
- The minecraft_genai runbook still says `/mnt/user/llm:/gguf`. The user is removing that runbook.
