#!/bin/sh
# p26/chain9.sh: the DSpark drafter under ZHA at 512K. ctx-ds14 (tower-ctx: the drafter's full layer in a 32K ring,
# a window's rows one at a time through the chunk cache, wrong guesses taken back without a jump) with --draft-model,
# with --window 4, and as is, against ctx-d200e08b, two rotation rounds, short prompt (800 tokens) and the 4,521-token
# prompt (400).
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
OLD=polystrata:ctx-d200e08b; NEW=polystrata:ctx-ds14
DM="--draft-model /gguf/inkling/dspark-Inkling-2-Mini-Q8_0_Old.gguf"
trap 'docker start PolyStrata >/dev/null; touch $D/p26/chain9.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p26/bench.sh
N=200 IMAGE=$NEW ARGS="$DM" $B c9-warm 524288 $Z
for r in "d200 a" "same a" "ds a" "w4 a" "w4 b" "ds b" "same b" "d200 b"; do
  set -- $r; I=$NEW; A=""
  [ $1 = d200 ] && I=$OLD; [ $1 = ds ] && A="$DM"; [ $1 = w4 ] && A="--window 4"
  N=800 ARGS="$A" IMAGE=$I $B c9-$1-$2-short 524288 $Z
  IDS=ids-4k.txt ARGS="$A" IMAGE=$I $B c9-$1-$2-mid 524288 $Z
done
