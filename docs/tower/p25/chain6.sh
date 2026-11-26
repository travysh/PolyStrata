#!/bin/sh
# p25/chain6.sh: speed rounds. Three rounds of the short prompt (800 tokens) on the live image, e3331820 and fd6d84bc,
# then three of the 4,521-token prompt (400) on the live image and fd6d84bc; engine by itself, ZHA 512K.
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
trap 'docker start PolyStrata >/dev/null; touch $D/p25/chain6.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p25/bench.sh
for k in 1 2 3; do
  N=800 IMAGE=polystrata:tower $B c6-live-short-$k 524288 $Z
  N=800 IMAGE=polystrata:ctx-e3331820 $B c6-e333-short-$k 524288 $Z
  N=800 IMAGE=polystrata:ctx-fd6d84bc $B c6-fd6d-short-$k 524288 $Z
done
for k in 1 2 3; do
  IDS=ids-4k.txt IMAGE=polystrata:tower $B c6-live-mid-$k 524288 $Z
  IDS=ids-4k.txt IMAGE=polystrata:ctx-fd6d84bc $B c6-fd6d-mid-$k 524288 $Z
done
