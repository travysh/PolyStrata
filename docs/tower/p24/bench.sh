#!/bin/sh
# p24/bench.sh LABEL MAXCTX [VAR=VALUE ...]: strata-poly by itself (no server) on the short prompt in p24/ids.txt,
# N (400) tokens at temperature 0, in image $IMAGE (polystrata:ctx-e20f5264). PRE: a command put in front (ncu ...).
# Live PolyStrata must be stopped first (one card).
D=/mnt/user/appdata/polystrata; O=$D/p24; L=$1; X=$2; shift 2
IMAGE=${IMAGE:-polystrata:ctx-e20f5264}
envs=""; for e in "$@"; do envs="$envs -e $e"; done
echo "== $L: $IMAGE ctx $X $* $(date)" > $O/$L.log
docker run --rm --gpus all --ulimit memlock=-1 -v /mnt/cache/llm:/gguf -v $D/data:/data -v $O:/p24 \
  -e STRATA_POLY_MIRROR=0 -e STRATA_CPU_NO_X4=1 $envs --entrypoint sh $IMAGE -c \
  "cd /opt/strata/terminal && $PRE engine-poly/strata-poly --model /gguf/inkling/Inkling-2-Mini-UD-Q4_K_XL.gguf \
   --max-context $X --expert-profile /data/packs/inkling-ud-q4_k_xl/expert-profile.txt --vram-reserve-mib 512 \
   --attn-requant q5_0 --loop-bf16 --tail-skip 4 --tokens \$(cat /p24/ids.txt) --max-new ${N:-400} --temperature 0 $ARGS" >> $O/$L.log 2>&1
