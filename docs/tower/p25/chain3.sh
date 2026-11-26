#!/bin/sh
# p25/chain3.sh: profile only. ZHA 512K's layer heads timed (image polystrata:ctx-time, STRATA_INK_TIME=1): the short
# prompt (800 tokens, the ring-only step) and the 4,521-token prompt (400 tokens, the full step), the full step also
# forced on the short prompt.
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
trap 'docker start PolyStrata >/dev/null; touch $D/p25/chain3.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p25/bench.sh
N=800 IMAGE=polystrata:ctx-time $B time-short 524288 $Z STRATA_INK_TIME=1
N=800 IMAGE=polystrata:ctx-time $B time-short-full 524288 $Z STRATA_INK_TIME=1 STRATA_INK_ZHA_SHORT=0
IDS=ids-4k.txt IMAGE=polystrata:ctx-time $B time-mid 524288 $Z STRATA_INK_TIME=1
N=800 IMAGE=polystrata:ctx-time $B notime-short 524288 $Z
