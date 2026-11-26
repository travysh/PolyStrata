#!/bin/sh
# p26/chain4.sh: ctx-lanes (ctx-poll, and Q5_0/Q8_0 rows a warp each in native_small_mmvq, NVFP4 expert rows 8 a warp)
# against ctx-poll in rotation rounds, short prompt (800 tokens) and the 4,521-token prompt (400); ctx-lanes also with
# each kernel switched off (STRATA_MMVQ_MANY_WARP=0, STRATA_NVFP4_LANES=0); a CUPTI trace of ctx-lanes (300).
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
P=polystrata:ctx-poll; X=polystrata:ctx-lanes; T=CUDA_INJECTION64_PATH=/p24/cupti_trace.so
trap 'docker start PolyStrata >/dev/null; touch $D/p26/chain4.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p26/bench.sh
N=200 IMAGE=$P $B c4-warm 524288 $Z
for r in "$P c4-poll" "$X c4-lanes" "$X c4-nowarp STRATA_MMVQ_MANY_WARP=0" "$X c4-nolanes STRATA_NVFP4_LANES=0" \
         "$X c4-lanes-b" "$P c4-poll-b"; do
  set -- $r
  N=800 IMAGE=$1 $B $2-short 524288 $Z $3
  IDS=ids-4k.txt IMAGE=$1 $B $2-mid 524288 $Z $3
done
N=300 IMAGE=$X $B tr-lanes-short 524288 $Z $T CUPTI_OUT=/p24/tr-lanes-short.csv
