# Community benchmark on NVIDIA GeForce RTX 2070

Measured on 2026-11-07 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2070, 8 GiB VRAM
- CPU: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz (6 cores, 12 threads); installed RAM: 31.8 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 592.82; CUDA: 13.1
- PolyStrata: project 0.1.45, commit a82999fd; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: lfm2.5-24b-a2b-ud-q4_k_xl
- Context: 65516 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly.exe --model LFM2.5-24B-A2B-UD-Q4_K_XL.gguf --max-context 65516 --expert-profile expert-profile.txt --conversation-cache-mib 8192 --prompt-batch 2048
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
| short chat | 39 (36-45) | 0 | 100 (100-100) | 9 | not measured | 28.1 (23.6-35.9) | 0.75 (0.64-1.20) |
| 4K prompt, fresh | 4126 (4126-4126) | 0 (0-0) | 24 cap | 3 | 984.5 (976.6-988.9) | not measured | 4.26 (4.24-4.28) |
| 16K prompt, fresh | 16490 (16490-16490) | 0 (0-0) | 24 cap | 3 | 1770.6 (1757.5-1796.0) | not measured | 9.38 (9.24-9.46) |
| agent session, follow-up turns | 14349 (14313-14359) at the last turn | see below | up to 64 | 3 | see below | 34.5 (24.8-44.0) | 3.70 (3.11-4.05) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 67.7 % (67.7-67.7); earlier-turn tokens read again per session: 299 (276-312) (earlier turns that the cache did not hold; the 13676 (13666-13679) new tokens of the files are read once); first turn (nothing cached) first token: 1.89 s (1.89-2.08).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 0 | 396 | 211.3 | 1.894 | 22.6 |
| 1 | 3654 | 391 | 3263 | 882.8 | 3.751 | 24.8 |
| 2 | 5313 | 3649 | 1664 | 538.3 | 3.125 | 27.4 |
| 3 | 8479 | 5308 | 3171 | 836.2 | 3.843 | 28.2 |
| 4 | 11506 | 8474 | 3032 | 848.3 | 3.629 | 32.0 |
| 5 | 14359 | 11501 | 2858 | 711.9 | 4.05 | 33.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 218 s.

## Notes on this report

- Machine: a laptop with an NVIDIA RTX 2070 mobile (8 GB), an Intel Core i7-10750H, 31.8 GiB of RAM and Windows 11; the settings above give the driver and engine versions.
- Measured on 2026-11-07 with `tools/strata_bench.py`: one suite run (three repeats per test) of this model with the install's own settings, thinking off. The run was not repeated, so small gaps between reports are within noise.
- This is not a result of the standard benchmark protocol; it was made before the protocol existed and does not implement it. Answer quality was not measured.
- The numbers and `results.json` are exactly as the tool wrote them; only this note was added when the report was prepared for submission.
