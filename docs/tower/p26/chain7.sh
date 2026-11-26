#!/bin/sh
# p26/chain7.sh: ctx-d200e08b with --window 2 and 4 (guesses from repeats of the conversation, --lookup) against
# --window 1, two rotation rounds, short prompt (800 tokens) and the 4,521-token prompt (400).
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
N1=polystrata:ctx-d200e08b
trap 'docker start PolyStrata >/dev/null; touch $D/p26/chain7.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p26/bench.sh
N=200 IMAGE=$N1 $B c7-warm 524288 $Z
for r in "1 a" "2 a" "4 a" "4 b" "2 b" "1 b"; do
  set -- $r
  N=800 ARGS="--window $1" IMAGE=$N1 $B c7-w$1-$2-short 524288 $Z
  IDS=ids-4k.txt ARGS="--window $1" IMAGE=$N1 $B c7-w$1-$2-mid 524288 $Z
done
