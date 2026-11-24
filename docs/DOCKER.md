# PolyStrata in Docker

Run Qwen3.8-Flash-Next in a container. The image holds the Strata project's engine and its Terminal, built here from
`tools/docker`; PolyStrata's own models (strata-poly) are not in it, so for them use `START-HERE.bat` or `./setup.sh`
([INSTALL.md](INSTALL.md)). Install Docker, the host GPU driver and the GPU-access requirements in the table below.
NVIDIA and AMD images can download and prepare a model on first start; Intel Arc needs prepared model files. Keep the
data volume to reuse downloads and settings.

> **On this page:** [Which image](#which-image-for-which-card) · [One command per card](#one-command-per-card) ·
> [Docker Compose](#docker-compose) · [Settings](#settings) · [The API key](#the-api-key) · [Connect your apps](#connect-your-coding-apps) ·
> [Watch and stop](#watch-it-and-stop-it) · [Update](#updating) · [Troubleshooting](#troubleshooting) · [Podman](#podman-rootless-on-ubuntu) ·
> [Build it yourself](#build-it-yourself)

Docker on a **Mac** has no access to the Mac's graphics chip (no Metal): run `./setup.sh` on the Mac
([MAC.md](MAC.md)). Docker Desktop on **Windows** passes NVIDIA cards through
(WSL 2) but not Radeon or Arc cards; on Windows use START-HERE.bat for supported NVIDIA or AMD cards. Intel Arc has no native Windows engine; follow the Linux guide.

## Which image for which card

| Your graphics card | Image tag | Needs on the computer |
| --- | --- | --- |
| NVIDIA **RTX 20, 30, 40, 50** series (and the matching RTX A / Quadro cards) | `strata:cuda13` | NVIDIA driver **580 or newer**, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| NVIDIA **GTX 10 (Pascal), Tesla P100, V100 (Volta)**, or any NVIDIA card whose driver is **575 to 579** | `strata:cuda12` | NVIDIA driver **575 or newer**, the same toolkit |
| AMD **RX 9000** series and Radeon AI PRO R9700 (RDNA 4, gfx1200 / gfx1201) | `strata:rocm-gfx120x` | the `amdgpu` kernel driver (Linux 6.8+) |
| AMD **RX 7000** series (RDNA 3, gfx1100 / gfx1101 / gfx1102) | `strata:rocm-gfx110x` | the `amdgpu` kernel driver |
| AMD **Ryzen AI Max** ("Strix Halo", Radeon 8060S, gfx1151) | `strata:rocm-gfx1151` | the `amdgpu` kernel driver |
| Intel **Arc** A-series and B-series (including Arc Pro B70) | `strata:sycl` (build it first: [Build it yourself](#build-it-yourself)) | the `i915` or `xe` kernel driver; **prepared model files** (see [Intel](#intel-arc)) |

Not sure which NVIDIA image? Take `cuda13` when `nvidia-smi` shows driver 580 or newer and the card is an RTX; otherwise `cuda12`.
A wrong pick does no harm: the container starts, says in plain words that the engine does not fit this card, and stops.

PolyStrata does not publish ready-made images. Build the image for your card with [Build it yourself](#build-it-yourself)
first; the examples below use the tags in this table (`strata:cuda13` and so on), and you can give your image another
name and use that in their place.

## One command per card

Each command runs PolyStrata in the background, keeps your files in the volume `strata-data`, and listens only on this computer
(`127.0.0.1:8080`). Choose `FAMILY` and `MODEL`, and generate your own key. The examples select ISTA-DASLab originals explicitly. The first start downloads the model (tens of GB): watch it with
`docker logs -f strata`. Already have the model's GGUF files? Add `-e MODEL_PATH=/models -v /your/folder:/models:ro` and nothing is
downloaded for the model itself ([Files you already have](#files-you-already-have)).

**NVIDIA**

```sh
docker run -d --name strata --gpus all -p 127.0.0.1:8080:8080 \
  --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 \
  -v strata-data:/data -e FAMILY=qwen -e MODEL=IQ3_XXS -e API_KEY=$(openssl rand -hex 24) \
  strata:cuda13
```

**AMD**

```sh
docker run -d --name strata --device /dev/kfd --device /dev/dri \
  --group-add video --group-add $(getent group render | cut -d: -f3) -p 127.0.0.1:8080:8080 \
  --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 \
  -v strata-data:/data -e FAMILY=qwen -e MODEL=IQ3_XXS -e API_KEY=$(openssl rand -hex 24) \
  strata:rocm-gfx120x
```

**Intel Arc**

```sh
docker run -d --name strata --device /dev/dri --group-add $(getent group render | cut -d: -f3) \
  -p 127.0.0.1:8080:8080 --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 \
  -v strata-data:/data -e FAMILY=qwen -e MODEL=IQ3_S -e MODEL_PATH=/models -v /srv/gguf/IQ3_S:/models:ro \
  -e API_KEY=$(openssl rand -hex 24) strata:sycl
```

What the options are for: `--ulimit memlock=-1:-1` lets the engine pin the RAM it uses for the model, `--shm-size 2g` gives
it shared memory, and `--stop-timeout 120` gives the model two minutes to unload when you stop the container (Docker's
default 10 seconds is too short for a big model). Do not put a memory limit on the container: PolyStrata plans from the whole
computer's RAM and loads tens of GB.

To print the key you made: `echo` it before you start (it is in your shell history otherwise), or ask the running container:
`docker exec strata strata launch --show-key --preset docker`.

### Intel Arc

PolyStrata's downloader prepares NVIDIA and AMD cards. For an Arc card the container **uses prepared model files**: the model's
GGUF files in a folder you mount (`-v /srv/gguf/IQ3_S:/models:ro` and `-e MODEL_PATH=/models`), and the prepared pack and MTP
layer in the data volume (`/data/PolyStrata-data/packs/<model>` and `/data/PolyStrata-data/mtp/rt`, made by the Intel installer on
this or another computer, see [INTEL.md](INTEL.md)). Without them the container tells you which file is missing and stops.
Nothing is downloaded or prepared for Arc inside the container. Arc B-series cards (xe driver) stream the experts from the
GGUF files, so the GGUF folder should be on a fast SSD.

## Docker Compose

Ready-made files are in [docker/compose/](../tools/docker/compose): `docker-compose.nvidia.yml`, `docker-compose.amd.yml`,
`docker-compose.intel.yml` and `.env.example`. Copy the one for your card and its `.env.example`, edit, start:

```sh
cd tools/docker/compose
cp .env.example .env              # set FAMILY=qwen, MODEL=..., API_KEY=...
docker compose -f docker-compose.nvidia.yml up -d
docker compose -f docker-compose.nvidia.yml logs -f
```

The files name the Strata project's images (`ghcr.io/niko1221/strata`) as their default. PolyStrata does not publish images
there, so set `STRATA_IMAGE` in `.env` to the image you built (for example `strata:cuda13`).

They set what the commands above set: the GPU (`deploy.resources.reservations.devices` for NVIDIA, `/dev/kfd` and
`/dev/dri` plus the `video` and `render` groups for AMD, `/dev/dri` for Intel), `ulimits.memlock`, `shm_size`, a two-minute
`stop_grace_period`, the `strata-data` volume, and the port bound to `127.0.0.1`. To let other computers reach it, set
`STRATA_BIND=0.0.0.0` in `.env` and keep a real `API_KEY`.

## Settings

Everything is an environment variable (`-e NAME=value`, or in compose). They are the same questions `strata launch` asks on its
screens; leave out what you do not care about.

| Variable | What it does | Default |
| --- | --- | --- |
| `MODEL` | model size: `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `Q2_0`, `UD-Q4_K_XL`, ... ([MODELS.md](MODELS.md)) | PolyStrata picks what suits the card, and says so in the log |
| `FAMILY` | `qwen`, `unsloth`, `gyro`, `coder`, `swift`, ...; use `qwen` for the original ISTA-DASLab packs | installer recommendation; may differ by hardware |
| `MODEL_PATH` | GGUF files you already have: a folder or one `.gguf`, **inside the container** (mount it with `-v`, `:ro` is fine) | download |
| `ALLOW_DOWNLOAD` | `1`: go on and download the model although model files of the older layout were left unread in `/data/models` | stop |
| `CONTEXT` | how much text it remembers at once: `32k`, `64k`, `128k`, `256k` or a number | the largest that fits |
| `PARALLEL` | requests answered at the same time, 1 to 8 | fitted to the card |
| `VISION` | `off`, `on` (the model reads pictures), `gpu`, `cpu`; the picture-reader file (about 0.9 GB) is downloaded with the model | `off` |
| `KV` | conversation memory format: `int8`, `q4_0` (half the memory), `k8v4`, `fp16` | `int8` |
| `DRAFTING` | which drafter guesses ahead: `auto`, `mtp`, `dflash2` or `off` | `auto` |
| `DRAFT_VOCAB` | which tokens the drafter may guess: `cjk` (with Chinese, Japanese and Korean), `en`, `cyrillic`, `fr`, `latin` or `arabic` (each with English and code) | picked from the container's languages, else `cjk` |
| `GPU` / `GPUS` | card numbers as the monitor lists them, `0` or `0,1`; empty or `all` = the engine's choice. Changing it later makes the model's run settings again (the files stay) | all that fit |
| `GPU_ROLES` | a role for each card, `1=main,0=helper` (`main`, `helper`, `off`; cards you do not name are off; not together with `GPU` / `GPUS`). Passed to `launch --gpu-roles`; see [MULTI_GPU.md](MULTI_GPU.md) | none |
| `TUNE` | `quick` (runtime depends on hardware and workload), `thorough`, `skip`; done once and kept in the preset (`RETUNE=1` tunes again) | `skip` |
| `API_KEY` / `API_KEY_FILE` | the key clients must send ([below](#the-api-key)); the file form is for Docker secrets | made for you |
| `TUNE_ALLOW_SMALL` | `1` passes `--tune-allow-small` to `launch` (allow tuning on a small card) | `0` |
| `PORT` | the port inside the container | `8080` |
| `PRESET` | the preset name in `/data` | `docker` |
| `STRATA_SET` | advanced engine settings, space separated, like `strata launch --set` (`"prefill=65536 STRATA_PIN_BY_NEED=1"`) | none |

The route tail skip (skipping rarely used parts of the model) is off in the container, as everywhere. `STRATA_SET="STRATA_ROUTE_TAIL_SKIP=7"` turns it on;
the numbers for both choices are in [DETAILS.md](DETAILS.md#route-tail-skip-strata_route_tail_skip-off-by-default-since-0217).

Your choices are saved as a preset in the volume. Start again with no settings and the saved preset is used; start with
settings and they replace the saved ones (tuning results are kept). To change one thing, recreate the container with the new
variable; the model and tuning stay in `/data`. Anything else: `docker exec strata strata launch --help`, or
`docker run --rm IMAGE strata launch --plan --family qwen --model q2_0` to see what a choice would do without starting anything.

What is in the volume `/data`: `PolyStrata-data/` (model files, the prepared pack), `state/` (presets, tuning, the key), `install/`
(run settings and logs), `home/`. The engine and the Terminal are in the image, so updating the image never touches your files.

To keep it all on a big disk, mount a folder there instead of the named volume: `-v /mnt/big/strata:/data` (the folder must be writable by the
container's user). Nothing large is written anywhere else in the container.

### Files you already have

Mount the folder with the model's GGUF files read-only and name it: `-v /srv/gguf/IQ3_XXS:/models:ro -e MODEL_PATH=/models
-e FAMILY=qwen -e MODEL=IQ3_XXS`. Nothing is downloaded for the model itself; the prepared pack is made in the volume.

- The folder holds the shards of one model directly (`...-00001-of-00002.gguf`, `...-00002-of-00002.gguf`), not a folder of folders.
  If you mount a parent folder, the container says which sub-folder to name in `MODEL_PATH`.
- The folder has to be mounted: a `MODEL_PATH` that is not in the container, or a folder with no `.gguf` files, stops the start at
  once with a message, and nothing is downloaded.
- Read-only (`:ro`) is fine. What is written to the volume is small next to the model: measured with the IQ3_XXS files, the prepared pack
  is 1.5 GB (the engine reads the experts from your GGUF files), the DFlash2 drafter 0.8 GB on disk (a 1.2 GB download, fetched once), and
  with `VISION=on` the picture reader 0.9 GB: 3.1 GB for the whole volume. With the MTP draft layer as the drafter instead (`DRAFTING=mtp`) the
  layer is about 6 GB, fetched once. `VISION=off` skips the picture reader.
- Links to the files work when the links' targets are mounted too, and `MODEL` is set (without `MODEL`, PolyStrata identifies the files itself
  and does not follow links).
- Without `MODEL_PATH`, files you put into the volume by hand (`/data/PolyStrata-data/models/<model folder>/`) are found as well, if they have the
  published names.

## The API key

The server listens on every address **inside** the container (that is what lets the published port work), and PolyStrata does not
serve beyond this computer without a key. So a key is always on:

- **You set it** with `API_KEY` (16 characters or more; `openssl rand -hex 24` makes one) or `API_KEY_FILE`.
- **You do not**: the first start makes one, keeps it in the volume, and the log tells you how to show it:
  `docker exec strata strata launch --show-key --preset docker`.

Publish the port as `-p 127.0.0.1:8080:8080` to keep it on this computer. Publish it as `-p 8080:8080` (or bind `0.0.0.0`) only
when other computers should reach it, and then use a key you chose. PolyStrata warns in the log when the key is short. Every
request must send `Authorization: Bearer <key>`; `GET /health` answers without one.

## Connect your coding apps

From outside the container, the address is the published port and the key is yours:

- OpenAI-compatible apps: base URL `http://127.0.0.1:8080/v1`, API key = your key, model name from `GET /v1/models`.
- Anthropic-compatible apps (Claude Code): base URL `http://127.0.0.1:8080`, same key.

`strata connect` writes the settings file for OpenCode, Codex CLI, Claude Code, oh-my-pi, pi, Hermes Agent and OpenClaw, and
checks that a prompt and a tool call work. **Inside** the container it fills in the key and the model for you:

```sh
docker exec strata strata connect opencode          # prints the file and the environment
docker exec strata strata connect opencode --check  # one prompt and one tool call, the way that app sends them
```

It prints `http://127.0.0.1:8080`, which is the address *inside* the container; on your computer use the published address
(`-p 127.0.0.1:9000:8080` makes it `http://127.0.0.1:9000`). If your coding app runs on your computer, install the `strata`
Terminal there ([TERMINAL_LAUNCH.md](TERMINAL_LAUNCH.md)) and run `strata connect opencode --url http://HOST:8080 --key <your key>`.

## Watch it and stop it

```sh
docker logs -f strata                       # what it is doing (the first start: the download)
docker exec -it strata strata monitor       # the live monitor (q leaves it, the server keeps running)
docker exec strata strata monitor --once    # one snapshot, for scripts
docker ps                                   # "healthy" once the model is loaded; "starting" while it downloads or loads
docker stop strata                          # drains, unloads the model, frees the graphics memory, exits 0
docker start strata                         # serving again in about a minute: the model and your choices are in the volume
```

The health check asks `GET /v1/models` with the key. During the first download the container shows `starting`, not `unhealthy`.
`docker stop` sends SIGTERM; the container runs `strata launch --stop`, so the engine releases its memory properly. Give it time
(`--stop-timeout 120` or compose's `stop_grace_period: 2m`): with the default 10 seconds a big model can be killed (exit 137).

## Updating

```sh
# build the image again from the updated source or engine package (see Build it yourself)
docker stop strata && docker rm strata
# run the same docker run command again with the rebuilt image (compose: docker compose up -d --build)
```

The models, the prepared pack, your preset and tuning stay in the volume `strata-data`; nothing is downloaded again. A new
engine can make an old tuning result less useful: start once with `-e TUNE=quick -e RETUNE=1` to tune again.
`strata update` inside the container is not the way to update: the engine is part of the image.

## Troubleshooting

**"The engine in this image does not match this computer's graphics card"** or the container stops right at the start.
The card is not visible to the container, or the image is for another card. Check the options for your card in
[the commands](#one-command-per-card) and the table in [Which image](#which-image-for-which-card).

**NVIDIA: `could not select device driver "" with capabilities: [[gpu]]`, or `docker: Error response from daemon: unknown or
invalid runtime name: nvidia`.** The NVIDIA Container Toolkit is missing or Docker was not told about it:

```sh
sudo apt install nvidia-container-toolkit          # or your distribution's package; see NVIDIA's install guide
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi       # must list your card
```

With Podman use CDI: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`, then `--device nvidia.com/gpu=all`. If Podman
cannot read that file (Ubuntu 24.04's Podman 4.9), see [Podman](#podman-rootless-on-ubuntu) below.
If `nvidia-smi` inside the container works but PolyStrata says the driver is too old, you picked `cuda13` with a driver below 580:
use the `cuda12` image or update the driver.

**AMD: `Permission denied` on `/dev/kfd`, or no card found.** The container needs the device files and your user (or the
container's user) must be allowed to use them. Add `--device /dev/kfd --device /dev/dri` and the groups
`--group-add video --group-add $(getent group render | cut -d: -f3)` (the render group's number differs from computer to
computer, so use the number, not the name). `ls -l /dev/kfd /dev/dri/renderD*` shows who owns them. On a computer with
SELinux (Fedora, RHEL) that still refuses the device files, add `--security-opt label=disable` to the `docker run`.

**Intel: no device found.** Pass `--device /dev/dri` and `--group-add <render group number>` as above. An Arc card needs the
`i915` or `xe` driver on the host (`ls /dev/dri/renderD*` must show a device; `lspci -k | grep -A3 -i arc` shows the driver in use).

**SELinux (Fedora, RHEL, Rocky): `Permission denied` on a folder you mounted.** Add `:z` (shared) or `:Z` (private) to the
mount: `-v /srv/gguf/IQ3_XXS:/models:ro,z`. A named volume like `strata-data` needs nothing.

**The container is killed or the computer swaps.** PolyStrata loads tens of GB into RAM: leave the container without a memory
limit, close other big programs, and pick a smaller `MODEL` ([MODELS.md](MODELS.md)). The log prints a warning when it sees a
memory limit.

**`docker stop` took long or ended with exit code 137.** The model was still unloading when Docker's time ran out. Use
`--stop-timeout 120` (compose: `stop_grace_period: 2m`).

**Port already in use.** Publish another host port: `-p 127.0.0.1:9000:8080` (inside the container it stays 8080).

**The first start says it is downloading but nothing moves.** `docker logs -f strata` shows the percentage every few seconds;
an interrupted download continues where it stopped when you start the container again. The volume needs the model's size free
(the log names it).

**"No MODEL was set"** warning: fine, PolyStrata chose a model for your card; set `MODEL` to choose yourself.

### Podman (rootless, on Ubuntu)

The image runs under Podman as well; `docker` in the commands above becomes `podman`. Two things on such a computer differ from the Docker commands. Both are about the host, not
the image.

**`Error: setting up CDI devices: unresolvable CDI devices nvidia.com/gpu=all`.** Podman 4.9 reads the CDI file (`/etc/cdi/nvidia.yaml`),
but a newer `nvidia-ctk` writes a field it does not know (`unknown field "additionalGids"` in `podman --log-level=debug run ...`), so it
ignores the whole file. A newer Podman reads it. Without one, pass the card the way the toolkit would: the device files of your card (one
`--device /dev/nvidiaN` for each card, with the three shared ones) and the driver's libraries, read-only, at the names the engine
looks for:

```sh
LIB=/usr/lib/x86_64-linux-gnu          # where your computer keeps them: ldconfig -p | grep 'libcuda.so.1'
GPU="--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools"
for lib in libcuda libnvidia-ml libnvidia-ptxjitcompiler libnvidia-nvvm; do
  [ -e $LIB/$lib.so.1 ] && GPU="$GPU -v $(readlink -f $LIB/$lib.so.1):$LIB/$lib.so.1:ro"
done
GPU="$GPU -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro"
```

then use `$GPU` where the Docker commands have `--gpus all`:

```sh
podman run -d --name strata $GPU -p 127.0.0.1:8080:8080 \
  --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 \
  -v strata-data:/data -e FAMILY=qwen -e MODEL=IQ3_XXS -e API_KEY=$(openssl rand -hex 24) \
  strata:cuda12
```

If no card reaches the container, its start says so in plain words and stops (exit code 3) before anything is downloaded.

**The container is killed while the model loads (exit code 137, no error in `podman logs`, `OOMKilled` is false).** Ubuntu's
`systemd-oomd` ends a program of your login when its memory has been under strain for 20 seconds, and a rootless container lives below your
user manager (`user@<your user number>.service`), where it watches. Loading a model bigger than the free RAM can do that, most of all when an
earlier start left its copy of the model's files in the memory cache. `podman ps` can even go on saying
"running" for a container whose processes are gone; `journalctl --user | grep "oomd killed"` shows the kill.

- `--cgroups=disabled` is **not** a way out: Podman still puts its `conmon` and the container in a `podman-<number>.scope` below the user manager
  with that flag (checked in `/proc/<pid>/cgroup`), and a container started with it was ended by `systemd-oomd` too.
- Docker (rootful) puts containers in the system's slice, where `systemd-oomd` does not look by default; that case was not tested.

## Build it yourself

Build the image on a Linux x86-64 computer with Docker. Two ways, and both leave the image in the local Docker (nothing is pushed).

**From an engine package.** You need an engine package folder (made by `tools/packaging/package_engine.py`) and:

```sh
tools/docker/build.sh --flavor cuda --engine ~/pkgs/<cuda13-package-folder> --tag strata:cuda13
tools/docker/build.sh --flavor hip  --engine ~/pkgs/<hip-rx9070-package-folder> --tag strata:rocm-gfx120x
tools/docker/build.sh --flavor sycl --engine ~/pkgs/<sycl-package-folder> --tag strata:sycl
```

`build.sh` assembles the build context (the Terminal's files and the package) and runs `docker build`. The build makes the
Terminal as the release archive does (Python 3.12 and the locked packages, fetched at build time only), then checks the
engine package before it may enter the image: every file's SHA-256, and that every library it imports is bundled or belongs
to the system or the graphics driver (`tools/packaging/binary_deps.py`). Finally it checks that the dynamic linker resolves
every library inside the finished image. A package that fails any of these stops the build.
`tools/docker/smoke-test.sh` runs the checks on a real card (serves, key, chat, tool call, monitor, clean stop, memory freed,
restart); see its first lines for the variables.

**From source.** The CUDA 13 engine is compiled inside Docker, packaged, and put in the same kind of image. `.git` is not copied
into the build, so pass the commit in as `STRATA_COMMIT`:

```sh
docker build -t strata:src -f tools/docker/Dockerfile.src --build-arg STRATA_COMMIT=$(git rev-parse HEAD) .
```

`--build-arg CUDA_ARCHITECTURES=89` builds for one card only, which is much faster. For a GTX 10, P100 or V100 build the CUDA 12
variant: `docker build -t strata:src-cuda12 -f tools/docker/Dockerfile.src --build-arg STRATA_COMMIT=$(git rev-parse HEAD)
--build-arg CUDA_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04 --build-arg CUDA_ARCHITECTURES="60;61;70;75;80;86;89" .`
The picture reader is built for the same architectures as the engine; `--build-arg BUILD_VISION=0` leaves it out for a text-only
package. The build takes its ignore list from `tools/docker/Dockerfile.src.dockerignore`.
For AMD and Intel builds from source see [AMD_HIP.md](AMD_HIP.md) and [INTEL.md](INTEL.md); then package the result with
`tools/packaging/package_engine.py` and use `tools/docker/build.sh`.

## Known limits

- Linux x86-64 hosts only. Docker on Windows reaches NVIDIA cards only; Docker on a Mac cannot use the GPU.
- The images run as `root` inside the container (GPU device files, page-locked memory). Run the container on a computer you
  trust the way you would run the app; do not mount folders you do not want it to read.
- An Arc card needs prepared model files (see [Intel](#intel-arc)).
- AMD **RX 7000** (gfx110x) and RDNA 3 iGPUs: the gfx110X image is built for gfx1100 / 1101 / 1102 (RX 7900, 7800, 7700, 7600). The
  Radeon 780M (gfx1103) is experimental and has its own engine package, not in these images.

