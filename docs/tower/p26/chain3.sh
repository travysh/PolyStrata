#!/bin/sh
# p26/chain3.sh: ctx-poll (ctx-defer, and the rings already rung answered between the layers' launches) against
# fd6d84bc and ctx-defer in rotation rounds, short prompt (800 tokens) and the 4,521-token prompt (400); a CUPTI trace
# of ctx-poll on the short prompt (300).
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
P=polystrata:ctx-poll; F=polystrata:ctx-fd6d84bc; Q=polystrata:ctx-defer; T=CUDA_INJECTION64_PATH=/p24/cupti_trace.so
trap 'docker start PolyStrata >/dev/null; touch $D/p26/chain3.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p26/bench.sh
N=200 IMAGE=$F $B c3-warm 524288 $Z
for r in "$F c3-fd6d" "$P c3-poll" "$Q c3-defer" "$P c3-poll-b" "$F c3-fd6d-b" "$Q c3-defer-b"; do
  set -- $r
  N=800 IMAGE=$1 $B $2-short 524288 $Z
  IDS=ids-4k.txt IMAGE=$1 $B $2-mid 524288 $Z
done
N=300 IMAGE=$P $B tr-poll-short 524288 $Z $T CUPTI_OUT=/p24/tr-poll-short.csv
