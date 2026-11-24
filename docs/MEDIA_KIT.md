# PolyStrata media and video guide

For people and AI tools creating an article, overview or video. Last reviewed: 20 November 2026, for PolyStrata 0.2.20.
Use [AI_REFERENCE.md](AI_REFERENCE.md) for product facts and [README.md](../README.md) for the public introduction.

## Short description

PolyStrata runs large AI models on your own computer: Qwen3.8-Flash-Next on the Strata project's engine, and eleven
more models (GLM 5.3 Flash, DeepSeek V4 Flash, Qwen3.6 and others) on an engine written for PolyStrata. It is installed
from a copy of the repository with setup (`START-HERE.bat` on Windows, `./setup.sh` on Linux and on a Mac) and used
through its OpenAI- and Anthropic-compatible API: from an app such as Linger, a coding tool or the terminal. PolyStrata's code is free and open
source under the MIT license.

Suggested opening: “Run large AI models on your own PC.”

## A short video outline

1. **Show it in an app.** Once setup has started a model, connect [Linger](https://github.com/simplexai-labs/linger) or
   another app to it. A chat and a visible result explain the product quickly. `http://127.0.0.1:8080` has a live
   monitor of the model and the GPU, CPU and RAM, and the request monitor.
2. **Show the two engines.** Qwen3.8-Flash-Next runs on the Strata project's engine; the eleven other models run on
   PolyStrata's own engine and are experimental. Explain the [availability limits](AI_REFERENCE.md#model-families).
3. **Choose a model for the machine.** For Qwen3.8-Flash-Next, prefer ISTA-DASLab, then Unsloth, Gyro and other
   providers where compatible ([provider preference](AI_REFERENCE.md#provider-preference-qwen38-flash-next)).
   Gyro is a memory-saving option, currently slower. State the model, quantization and hardware used in any live demo.
4. **Connect a familiar tool.** Claude Code, Codex CLI, OpenCode or any OpenAI-compatible app
   ([connecting apps](DETAILS.md#connecting-apps)). Show a real successful connection if demonstrating one.
5. **Close with the repository:** [simplexai-labs/PolyStrata](https://github.com/simplexai-labs/PolyStrata).

This is an outline, not evidence that any particular model, workflow or benchmark was tested for the video.

## Existing assets

Use the existing assets with accurate captions. PolyStrata has no logo file; write its name as text. Some pictures
come from the Strata project and show its Terminal or its demo, which this repository does not publish: caption
them as the Strata project's, not as PolyStrata. The Linger picture is the Linger project's.

| Asset | File | Suggested caption / context |
| --- | --- | --- |
| Speed chart | [speed.svg](media/speed.svg) | Tokens per second while writing an answer in a short chat, for the models of PolyStrata's own engine, on a laptop with an RTX 4070 Laptop (8 GB) and on an RTX 5090. Keep the card and model with any number taken from it. |
| Coding agent demo | [claude-code-demo.gif](media/claude-code-demo.gif) | Claude Code on Qwen3.6-35B-A3B UD-Q3_K_M with PolyStrata's API monitor beside it, on a laptop with an 8 GB card, in real time. |
| Engine explanation | [how-it-works.svg](media/how-it-works.svg), [guess-and-check.svg](media/guess-and-check.svg) and the per-model versions in [media/](media/) | Illustrations of each documented engine path, not proof that every model uses the same path. |
| The Strata project's coding demo | [pagoda-preview.webp](media/pagoda-preview.webp), [Pagoda.mp4](https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4) | A voxel garden generated with the Strata project's release, documented with an RTX 5070, IQ3_S and 128K context. |
| Linger | [linger-desktop.jpg](media/linger-desktop.jpg) | Linger, the app the README recommends, on a Mac: agents report in a chat room beside the Chrome windows they work in. Copied unchanged from the [Linger project](https://github.com/simplexai-labs/linger) (`assets/feature-desktop-three-agents.jpg`), under [Linger's license](https://github.com/simplexai-labs/linger/blob/main/LICENSE) (Apache 2.0 with added conditions). |
| The Strata project's Terminal | [terminal-monitor.png](media/terminal-monitor.png) | Its monitor during a Gyro-S session; the live counters are not a standardized benchmark. |

## Accuracy rules

- Use actual footage or label illustrations. Do not invent a successful install, generated answer or speed reading.
- Say which engine a model runs on, and call the models of PolyStrata's own engine experimental, as the README does.
- Keep the provider preference distinct from measured performance. The Strata project maintainer's collaboration
  with ISTA-DASLab explains the recommendation order; it does not prove a benchmark result.
- Do not promise “any PC,” “every model,” or identical results across GPUs. Follow the model and hardware guides.
- For speed comparisons, use a source report with hardware, model, quantization, engine version and workload.
  Keep reasoning tokens, decode speed, prompt speed and time to first answer distinct.
- Local inference, internet downloads and connected tools are separate. Refer to [SAFETY.md](SAFETY.md) for privacy claims.
- PolyStrata's MIT license does not replace the licenses of model weights, dependencies or other creators' work.
  Use the [credits and licensing guide](HOW_IT_WORKS.md#credits).

## Links

- Project: [simplexai-labs/PolyStrata](https://github.com/simplexai-labs/PolyStrata)
- Questions and reports: [issues](https://github.com/simplexai-labs/PolyStrata/issues)
- The Strata project, whose engine runs Qwen3.8-Flash-Next here: [Niko1221/Strata](https://github.com/Niko1221/Strata)
