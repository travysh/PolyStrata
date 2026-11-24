# Community benchmark on NVIDIA GeForce RTX 4060 Ti

Measured on 2026-11-22 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4060 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 8 threads); installed RAM: 30.9 GiB; storage: NVMe CT1000P2SSD8 SSD/NVMe; WD_BLACK SN7100 1TB SSD/NVMe
- OS: Windows 11 (10.0.26300); GPU driver: 621.24; CUDA (driver supports up to): 13.5; engine built for CUDA 13.0
- PolyStrata: main at d1f711e11 (a linked worktree; the tool could not read the commit), ready-made engine build 605522f2; engine 0.1.65
- GPU detail (after the run, near idle): power_limit_w=165.00, power_draw_w=104.04, pcie=gen4 x2, pstate=P2, temp_c=59.39
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: inkling-2-mini-ud-q4_k_xl
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly.exe --model Inkling-2-Mini-UD-Q4_K_XL.gguf --max-context 32768 --expert-profile expert-profile.txt --conversation-cache-mib auto
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a PolyStrata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 44 (43-50) | 0 | 100 (100-100) | 9 | not measured | 39.3 (31.3-42.6) | 1.03 (0.95-1.19) |
| 4K prompt, fresh | 4075 (4075-4075) | 0 (0-0) | 24 cap | 3 | 662.1 (662.1-663.0) | not measured | 6.17 (6.17-6.19) |
| 16K prompt, fresh | 16128 (16128-16128) | 0 (0-0) | 24 cap | 3 | 870.5 (870.3-875.6) | not measured | 18.55 (18.46-18.55) |
| agent session, follow-up turns | 11651 (11651-11654) at the last turn | see below | up to 64 | 3 | see below | 33.1 (31.8-34.8) | 4.86 (4.65-5.09) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.9 % (69.9-70.0); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 10813 (10813-10813) new tokens of the files are read once); first turn (nothing cached) first token: 2.24 s (2.20-2.39).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 555 | 0 | 555 | 234.2 | 2.388 | 30.1 |
| 1 | 2743 | 581 | 2162 | 449.2 | 4.843 | 32.1 |
| 2 | 4970 | 2806 | 2164 | 453.4 | 4.821 | 32.8 |
| 3 | 7197 | 5033 | 2164 | 426.5 | 5.093 | 32.1 |
| 4 | 9424 | 7260 | 2164 | 447.8 | 4.862 | 34.6 |
| 5 | 11651 | 9487 | 2164 | 443.4 | 4.906 | 33.1 |

### Skipped and failed

- skipped 32K: the context is 32768 tokens

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 238 s.
