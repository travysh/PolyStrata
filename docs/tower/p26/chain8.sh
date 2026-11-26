#!/bin/sh
# p26/chain8.sh: does the DSpark drafter pay on Tower? ctx-d200e08b at a 64K context without ZHA (the drafter refuses
# ZHA), with and without --draft-model, two rotation rounds, short prompt (800 tokens) and the 4,521-token one (400).
D=/mnt/user/appdata/polystrata; N1=polystrata:ctx-d200e08b
DM="--draft-model /gguf/inkling/dspark-Inkling-2-Mini-Q8_0_Old.gguf"
trap 'docker start PolyStrata >/dev/null; touch $D/p26/chain8.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p26/bench.sh
N=200 IMAGE=$N1 $B c8-warm 65536
for r in "off a" "ds a" "ds b" "off b"; do
  set -- $r; A=""; [ $1 = ds ] && A="$DM"
  N=800 ARGS="$A" IMAGE=$N1 $B c8-$1-$2-short 65536
  IDS=ids-4k.txt ARGS="$A" IMAGE=$N1 $B c8-$1-$2-mid 65536
done
