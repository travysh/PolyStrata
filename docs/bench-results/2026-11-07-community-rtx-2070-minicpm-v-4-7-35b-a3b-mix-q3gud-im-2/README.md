# Community benchmark on NVIDIA GeForce RTX 2070

Measured on 2026-11-07 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2070, 8 GiB VRAM
- CPU: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz (6 cores, 12 threads); installed RAM: 31.8 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 592.82; CUDA: 13.1
- PolyStrata: project 0.1.45, commit a82999fd; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: minicpm-v-4.7-35b-a3b-mix_q3gud_im
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly.exe --model MiniCPM-V-4_7-35B-A3B-mix_q3gud_im.gguf --max-context 65536 --expert-profile expert-profile.txt
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 44.7 (36.8-47.3) | 0.69 (0.57-1.00) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 751.4 (745.6-760.5) | not measured | 5.38 (5.33-5.42) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1029.8 (1021.9-1034.8) | not measured | 15.74 (15.64-15.95) |
| agent session, follow-up turns | 15027 (15027-15029) at the last turn | see below | up to 64 | 3 | see below | 43.5 (36.0-47.9) | 5.93 (3.90-6.23) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.1 % (69.1-69.1); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 14201 (14201-14201) new tokens of the files are read once); first turn (nothing cached) first token: 2.24 s (2.22-2.61).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 234.5 | 2.613 | 37.5 |
| 1 | 4045 | 619 | 3426 | 614.6 | 6.145 | 36.0 |
| 2 | 5691 | 4094 | 1597 | 396.9 | 4.08 | 39.3 |
| 3 | 9121 | 5749 | 3372 | 550.0 | 6.167 | 41.2 |
| 4 | 12085 | 9163 | 2922 | 563.0 | 5.852 | 41.0 |
| 5 | 15027 | 12138 | 2889 | 486.4 | 6.026 | 41.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 247 s.

## Notes on this report

- Machine: a laptop with an NVIDIA RTX 2070 mobile (8 GB), an Intel Core i7-10750H, 31.8 GiB of RAM and Windows 11; the settings above give the driver and engine versions.
- Measured on 2026-11-07 with `tools/strata_bench.py`: one suite run (three repeats per test) of this model with the install's own settings, thinking off. The run was not repeated, so small gaps between reports are within noise.
- Context: 65,536 tokens; every test ran. (A separate report at 8,192 tokens skipped the 16K and agent tests.)
- This is not a result of the standard benchmark protocol; it was made before the protocol existed and does not implement it. Answer quality was not measured.
- The numbers and `results.json` are exactly as the tool wrote them; only this note was added when the report was prepared for submission.
