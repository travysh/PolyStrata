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

**To make it live** (not done): `docker tag polystrata:p19-mmq polystrata:tower`, then `rebuild_container PolyStrata`,
`docker start PolyStrata`. The template already names `polystrata:tower`. Rollback: tag
`polystrata:tower-c868922e` back. For the next image build, add the two `-D` options to `p18/Dockerfile`'s engine step.
`setup.build_engine_poly` passes `STRATA_GGML_DIR` from the environment but has no switch for `STRATA_POLY_MMQ`, so
the build has to call CMake itself, as `p19/Dockerfile` does.

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
