#!/bin/sh
# p26/chain13.sh: ctx-ds16 (3e3dc7514 and the drafter's scratch lent with the prompt's: its features a layout's rows
# apart) with the drafter against ctx-d200e08b, two rotation rounds at 512K with ZHA, short (800) and 4,521 tokens
# (400); then probe.sh's short chats and the 128K needle run through the server with the drafter config.
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
OLD=polystrata:ctx-d200e08b; NEW=polystrata:ctx-ds16
DM="--draft-model /gguf/inkling/dspark-Inkling-2-Mini-Q8_0_Old.gguf"
trap 'docker rm -f polystrata-next >/dev/null 2>&1; docker start PolyStrata >/dev/null; touch $D/p26/chain13.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p26/bench.sh
N=200 IMAGE=$NEW ARGS="$DM" $B c13-warm 524288 $Z
for r in "d200 a" "ds a" "ds b" "d200 b"; do
  set -- $r; I=$NEW; A=""
  [ $1 = d200 ] && I=$OLD; [ $1 = ds ] && A="$DM"
  N=800 ARGS="$A" IMAGE=$I $B c13-$1-$2-short 524288 $Z
  IDS=ids-4k.txt ARGS="$A" IMAGE=$I $B c13-$1-$2-mid 524288 $Z
done
IMAGE=$NEW sh $D/p25/probe.sh ds16 strata-inkling2-mini-ds.json
docker stop PolyStrata >/dev/null 2>&1
IMAGE=$NEW TASKS=trim-tasks sh $D/p24/runm.sh inkling p26-trim-ds16 strata-inkling2-mini-ds.json
