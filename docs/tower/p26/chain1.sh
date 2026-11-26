#!/bin/sh
# p26/chain1.sh: CUPTI traces (cupti_trace.so) of the engine by itself on image ctx-fd6d84bc: ZHA 512K on the short
# prompt (300 tokens) and the 4,521-token prompt (200), plain 64K on the short prompt; one run without the tracer.
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
T=CUDA_INJECTION64_PATH=/p24/cupti_trace.so; I=polystrata:ctx-fd6d84bc
trap 'docker start PolyStrata >/dev/null; touch $D/p26/chain1.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p26/bench.sh
N=300 IMAGE=$I $B tr-zha-short 524288 $Z $T CUPTI_OUT=/p24/tr-zha-short.csv
N=300 IMAGE=$I $B tr-plain-short 65536 $T CUPTI_OUT=/p24/tr-plain-short.csv
N=200 IDS=ids-4k.txt IMAGE=$I $B tr-zha-mid 524288 $Z $T CUPTI_OUT=/p24/tr-zha-mid.csv
N=300 IMAGE=$I $B notr-zha-short 524288 $Z
