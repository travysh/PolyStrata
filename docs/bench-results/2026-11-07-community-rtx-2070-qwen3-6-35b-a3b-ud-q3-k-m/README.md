# Community benchmark on NVIDIA GeForce RTX 2070

Measured on 2026-11-07 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2070, 8 GiB VRAM
- CPU: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz (6 cores, 12 threads); installed RAM: 31.8 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 592.82; CUDA: 13.1
- PolyStrata: project 0.1.45, commit a82999fd; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q3_k_m
- Context: 65516 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly.exe --model Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --max-context 65516 --expert-profile expert-profile.txt --tail-skip 6
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a PolyStrata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 56.5 (45.2-66.0) | 0.72 (0.62-0.86) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 791.9 (790.0-795.3) | not measured | 5.10 (5.08-5.11) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1078.5 (1074.4-1091.6) | not measured | 14.91 (14.74-14.97) |
| agent session, follow-up turns | 15004 (14995-15026) at the last turn | see below | up to 64 | 3 | see below | 55.7 (41.7-64.9) | 5.63 (4.13-14.26) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.6); earlier-turn tokens read again per session: 249 (240-271) (earlier turns that the cache did not hold; the 14181 (14181-14181) new tokens of the files are read once); first turn (nothing cached) first token: 2.69 s (2.67-2.81).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 218.4 | 2.806 | 48.1 |
| 1 | 4056 | 602 | 3454 | 242.6 | 14.255 | 41.7 |
| 2 | 5691 | 4049 | 1642 | 199.9 | 8.257 | 56.0 |
| 3 | 9108 | 5684 | 3424 | 512.7 | 6.707 | 57.6 |
| 4 | 12065 | 9101 | 2964 | 566.0 | 5.261 | 58.3 |
| 5 | 14995 | 12058 | 2937 | 519.2 | 5.689 | 43.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 249 s.

## Notes on this report

- Machine: a laptop with an NVIDIA RTX 2070 mobile (8 GB), an Intel Core i7-10750H, 31.8 GiB of RAM and Windows 11; the settings above give the driver and engine versions.
- Measured on 2026-11-07 with `tools/strata_bench.py`: one suite run (three repeats per test) of this model with the install's own settings, thinking off. The run was not repeated, so small gaps between reports are within noise.
- This is not a result of the standard benchmark protocol; it was made before the protocol existed and does not implement it. Answer quality was not measured.
- The numbers and `results.json` are exactly as the tool wrote them; only this note was added when the report was prepared for submission.
