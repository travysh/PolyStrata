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

**Tower's config is now 256K + `--kv q8_0`** (2026-10-17). The 64K fp16 original is
`/data/config/strata-cybertiel-ud-q4_k_xl.json.orig`: copy it back and restart to undo. A container restart keeps
the edited config. The container's `CONTEXT=65536` / `KV=int8` env are not what it runs. If the container is
recreated, check the engine's start line in the log.

### Next

1. Upstream: setup could pass `--kv q8_0` for the poly families when `KV=int8` (today it is silently ignored).
2. The CPU-path levers: a smaller quant (check whether the Cyber repo has UD-Q3_K_XL / UD-IQ3_XXS like Tiel's), or
   profiling the CPU expert path on Zen+.
3. Fork and push the `cybertiel-family` branch (the family commit only; these notes stay local).

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

## Cleanup still open

- `/mnt/disk2/llm/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` and `/mnt/disk2/llm/mmproj-F16.gguf`: duplicates of the cache
  copies. They are not kept, so delete them by hand.
- `strata-cybertiel` and `strata` both map port 8080: stop one before starting the other.
- `docker rm strata-old` once the SSD-backed `strata` is trusted.
- The minecraft_genai runbook still says `/mnt/user/llm:/gguf`. The user is removing that runbook.
