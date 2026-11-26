#!/bin/sh
# p26/chain5.sh: ctx-lanes as is, with the experts' split frozen (STRATA_FREEZE_EXPERTS=1: no swaps beside the CPU's
# experts), and with 6 and 11 CPU threads (8 is the default), short prompt (800 tokens) and the 4,521-token one (400).
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
X=polystrata:ctx-lanes
trap 'docker start PolyStrata >/dev/null; touch $D/p26/chain5.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p26/bench.sh
N=200 IMAGE=$X $B c5-warm 524288 $Z
for r in "c5-base| |" "c5-freeze| |STRATA_FREEZE_EXPERTS=1" "c5-t6|--threads 6|" "c5-t11|--threads 11|" "c5-base-b| |"; do
  l=${r%%|*}; rest=${r#*|}; a=${rest%%|*}; e=${rest#*|}
  N=800 ARGS="$a" IMAGE=$X $B $l-short 524288 $Z $e
  IDS=ids-4k.txt ARGS="$a" IMAGE=$X $B $l-mid 524288 $Z $e
done
