#!/bin/sh
# p26/chain6.sh: tower-ctx d200e08bc (fd6d84bc, the rings answered between the layers, NVFP4 expert rows 8 a warp)
# against fd6d84bc and the live image in rotation rounds, short prompt (800 tokens) and the 4,521-token prompt (400);
# d200e08bc with 11 CPU threads; 1,500 tokens; probe.sh's short chats; Pass 24's 128K needle run with live's config.
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
N1=polystrata:ctx-d200e08b; F=polystrata:ctx-fd6d84bc; L=polystrata:tower
trap 'docker start PolyStrata >/dev/null; touch $D/p26/chain6.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p26/bench.sh
N=200 IMAGE=$L $B c6-warm 524288 $Z
for r in "$L c6-live" "$F c6-fd6d" "$N1 c6-d200" "$N1 c6-d200-b" "$F c6-fd6d-b" "$L c6-live-b"; do
  set -- $r
  N=800 IMAGE=$1 $B $2-short 524288 $Z
  IDS=ids-4k.txt IMAGE=$1 $B $2-mid 524288 $Z
done
N=800 ARGS="--threads 11" IMAGE=$N1 $B c6-d200-t11-short 524288 $Z
IDS=ids-4k.txt ARGS="--threads 11" IMAGE=$N1 $B c6-d200-t11-mid 524288 $Z
N=1500 IMAGE=$N1 $B c6-d200-1500 524288 $Z
IMAGE=$N1 sh $D/p25/probe.sh d200 strata-inkling2-mini.json
docker stop PolyStrata >/dev/null 2>&1
IMAGE=$N1 TASKS=trim-tasks sh $D/p24/runm.sh inkling p26-trim-d200 strata-inkling2-mini.json
