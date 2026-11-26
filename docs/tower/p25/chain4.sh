#!/bin/sh
# p25/chain4.sh: profile only. zha_attend's answer step timed a kernel at a time (image polystrata:ctx-time2,
# STRATA_INK_TIME=1): the short prompt (800 tokens) with the ring-only step and with the full step, the 4,521-token
# prompt (400 tokens, the full step).
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
trap 'docker start PolyStrata >/dev/null; touch $D/p25/chain4.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p25/bench.sh
N=800 IMAGE=polystrata:ctx-time2 $B kt-short 524288 $Z STRATA_INK_TIME=1
N=800 IMAGE=polystrata:ctx-time2 $B kt-short-full 524288 $Z STRATA_INK_TIME=1 STRATA_INK_ZHA_SHORT=0
IDS=ids-4k.txt IMAGE=polystrata:ctx-time2 $B kt-mid 524288 $Z STRATA_INK_TIME=1
N=800 IMAGE=polystrata:ctx-time2 $B nokt-short 524288 $Z
