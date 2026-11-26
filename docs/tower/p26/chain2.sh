#!/bin/sh
# p26/chain2.sh: ctx-defer (fd6d84bc with the doorbell's rings answered once a token, not a layer at a time) against
# fd6d84bc and the live image, engine by itself on the short prompt (800 tokens) and the 4,521-token prompt (400), in
# rotation rounds; a traced run of ctx-defer on the short prompt; the 1,500-token run; then Pass 24's 128K needle run.
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
N1=polystrata:ctx-defer; F=polystrata:ctx-fd6d84bc; L=polystrata:tower
trap 'docker start PolyStrata >/dev/null; touch $D/p26/chain2.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p26/bench.sh
for r in "$L c2-live" "$F c2-fd6d" "$N1 c2-defer" "$N1 c2-defer-b" "$F c2-fd6d-b" "$L c2-live-b"; do
  set -- $r
  N=800 IMAGE=$1 $B $2-short 524288 $Z
  IDS=ids-4k.txt IMAGE=$1 $B $2-mid 524288 $Z
done
N=1500 IMAGE=$N1 $B c2-defer-1500 524288 $Z
docker stop PolyStrata >/dev/null 2>&1
IMAGE=$N1 TASKS=trim-tasks sh $D/p24/runm.sh inkling p26-trim-defer strata-inkling2-mini.json
