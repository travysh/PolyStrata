# Community benchmark on NVIDIA RTX A5000 Laptop GPU

Measured on 2026-11-22 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA RTX A5000 Laptop GPU, 16 GiB VRAM
- CPU: Intel(R) Xeon(R) W-11855M CPU @ 3.20GHz (6 cores, 12 threads); installed RAM: 63.2 GiB; storage: ASolid USB Unspecified/USB; Msft Virtual Disk SSD/File Backed Virtual; NVMe Samsung SSD 970 EVO Plus 2TB SSD/RAID; NVMe KXG60ZNV1T02 NVMe KIOXIA 1024GB SSD/RAID
- OS: Windows 11 (10.0.26300); GPU driver: 621.14; CUDA (driver supports up to): 13.5; engine built for CUDA 13.0
- PolyStrata: project 0.1.65, commit not a git checkout; engine 0.1.65
- GPU detail (after the run, near idle): power_limit_w=[N/A], power_draw_w=108.23, pcie=gen4 x8, pstate=P0, temp_c=61.41
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: gemma-4-26b-a4b-ud-q3_k_m
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly.exe --model gemma-4-26B-A4B-it-UD-Q3_K_M.gguf --max-context 65536 --expert-profile expert-profile.txt --draft-model mtp-gemma-4-26B-A4B-it-Q8_0.gguf --conversation-cache-mib auto
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
| short chat | 48 (45-53) | 0 | 100 (100-100) | 9 | not measured | 103.6 (95.9-135.6) | 0.16 (0.14-0.20) |
| 4K prompt, fresh | 4101 (4101-4101) | 0 (0-0) | 24 cap | 3 | 2532.6 (2528.7-2582.0) | not measured | 1.65 (1.64-1.65) |
| 16K prompt, fresh | 16390 (16390-16390) | 0 (0-0) | 24 cap | 3 | 2515.2 (2506.6-2526.0) | not measured | 6.55 (6.53-6.84) |
| 32K prompt, fresh | 32774 (32774-32774) | 0 (0-0) | 24 cap | 3 | 2264.6 (2259.2-2276.3) | not measured | 14.52 (14.49-15.15) |
| agent session, follow-up turns | 8438 (8438-8446) at the last turn | see below | up to 64 | 3 | see below | 163.2 (86.7-171.1) | 0.82 (0.69-0.87) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.0 % (69.0-69.0); earlier-turn tokens read again per session: 313 (313-321) (earlier turns that the cache did not hold; the 7764 (7764-7764) new tokens of the files are read once); first turn (nothing cached) first token: 0.28 s (0.27-0.39).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 0 | 396 | 1086.1 | 0.39 | 117.0 |
| 1 | 1970 | 389 | 1581 | 2193.1 | 0.761 | 94.4 |
| 2 | 3587 | 1963 | 1624 | 2042.0 | 0.819 | 155.0 |
| 3 | 5204 | 3580 | 1624 | 1997.0 | 0.84 | 166.8 |
| 4 | 6821 | 5197 | 1624 | 2051.0 | 0.801 | 163.2 |
| 5 | 8438 | 6814 | 1624 | 1936.1 | 0.853 | 159.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 107 s.

## Notes on this report

- Machine: a Dell Precision 7760 laptop with an NVIDIA RTX A5000 Laptop GPU (16 GB), an Intel Xeon W-11855M (AVX-512), 63 GB of DDR4-3200 RAM in two modules and Windows 11, plugged in. The settings above give the driver and engine versions.
- Build: PolyStrata commit d1f711e11 (2026-11-22), engine 0.1.65 (strata-poly). These numbers are from that build; an older report of this machine with engine 0.1.45 is not part of this submission.
- Measured with `tools/strata_bench.py`: one suite run (three repeats per test), thinking off, a freshly started server, context 65,536 tokens (the same as the RTX 2070 reports of 2026-11-07 for these models, which makes the two comparable).
- The PC was kept awake during the run: an earlier attempt was stalled by Windows Modern Standby and discarded. This run was checked for standby events and failed requests: none.
- The Windows page file was set to a fixed 64 GB. A laptop cooler was in use; GPU temperature and clocks were not logged during the runs (a spot check at low load read 47 C with no throttling).
- This is not a result of the standard benchmark protocol; it does not implement it. Answer quality was not measured.
- The numbers and `results.json` are exactly as the tool wrote them; only this note was added when the report was prepared for submission.
