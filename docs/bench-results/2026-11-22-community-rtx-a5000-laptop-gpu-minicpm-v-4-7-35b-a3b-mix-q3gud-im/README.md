# Community benchmark on NVIDIA RTX A5000 Laptop GPU

Measured on 2026-11-22 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA RTX A5000 Laptop GPU, 16 GiB VRAM
- CPU: Intel(R) Xeon(R) W-11855M CPU @ 3.20GHz (6 cores, 12 threads); installed RAM: 63.2 GiB; storage: ASolid USB Unspecified/USB; Msft Virtual Disk SSD/File Backed Virtual; NVMe Samsung SSD 970 EVO Plus 2TB SSD/RAID; NVMe KXG60ZNV1T02 NVMe KIOXIA 1024GB SSD/RAID
- OS: Windows 11 (10.0.26300); GPU driver: 621.14; CUDA (driver supports up to): 13.5; engine built for CUDA 13.0
- PolyStrata: project 0.1.65, commit not a git checkout; engine 0.1.65
- GPU detail (after the run, near idle): power_limit_w=[N/A], power_draw_w=112.24, pcie=gen4 x8, pstate=P0, temp_c=61.56
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: minicpm-v-4.7-35b-a3b-mix_q3gud_im
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly.exe --model MiniCPM-V-4_7-35B-A3B-mix_q3gud_im.gguf --max-context 65536 --expert-profile expert-profile.txt --conversation-cache-mib auto
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a PolyStrata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 103.8 (100.1-106.4) | 0.20 (0.20-0.22) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2414.5 (2392.7-2455.4) | not measured | 1.68 (1.65-1.70) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2598.8 (2598.7-2617.2) | not measured | 6.20 (6.15-6.22) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2447.3 (2444.6-2459.3) | not measured | 13.12 (13.06-13.14) |
| agent session, follow-up turns | 9085 (9085-9085) at the last turn | see below | up to 64 | 3 | see below | 103.9 (89.6-133.8) | 1.05 (0.90-1.15) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 71.0 % (71.0-71.0); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 8209 (8209-8209) new tokens of the files are read once); first turn (nothing cached) first token: 0.58 s (0.58-0.62).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 1028.5 | 0.622 | 105.4 |
| 1 | 2261 | 619 | 1642 | 1652.7 | 1.015 | 102.2 |
| 2 | 3967 | 2324 | 1643 | 1875.8 | 0.9 | 89.6 |
| 3 | 5673 | 4030 | 1643 | 1482.5 | 1.137 | 102.8 |
| 4 | 7379 | 5736 | 1643 | 1499.8 | 1.123 | 105.2 |
| 5 | 9085 | 7442 | 1643 | 1618.9 | 1.039 | 114.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 108 s.

## Notes on this report

- Machine: a Dell Precision 7760 laptop with an NVIDIA RTX A5000 Laptop GPU (16 GB), an Intel Xeon W-11855M (AVX-512), 63 GB of DDR4-3200 RAM in two modules and Windows 11, plugged in. The settings above give the driver and engine versions.
- Build: PolyStrata commit d1f711e11 (2026-11-22), engine 0.1.65 (strata-poly). These numbers are from that build; an older report of this machine with engine 0.1.45 is not part of this submission.
- Measured with `tools/strata_bench.py`: one suite run (three repeats per test), thinking off, a freshly started server, context 65,536 tokens (the same as the RTX 2070 reports of 2026-11-07 for these models, which makes the two comparable).
- The vision encoder runs on the GPU (setup's --vision gpu, up to 1,024 image tokens per picture); no pictures were sent in this suite.
- The PC was kept awake during the run: an earlier attempt was stalled by Windows Modern Standby and discarded. This run was checked for standby events and failed requests: none.
- The Windows page file was set to a fixed 64 GB. A laptop cooler was in use; GPU temperature and clocks were not logged during the runs (a spot check at low load read 47 C with no throttling).
- This is not a result of the standard benchmark protocol; it does not implement it. Answer quality was not measured.
- The numbers and `results.json` are exactly as the tool wrote them; only this note was added when the report was prepared for submission.
