#!/bin/sh
# p25/chain1.sh: the engine by itself on the short prompt: plain 64K and ZHA 512K on the live image, ZHA 512K on the
# test image with the ring-only step on and off (STRATA_INK_ZHA_SHORT=0), twice, 800 tokens; then 1,500 tokens on
# the test image both ways (the answer passes position 1,087, where the full step takes over).
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
T=${T:-polystrata:ctx-e3331820}
trap 'docker start PolyStrata >/dev/null; touch $D/p25/chain1.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
cp $D/p24/ids.txt $D/p25/ids.txt
B=$D/p25/bench.sh
for k in a b; do
  N=800 IMAGE=polystrata:tower $B plain-64k-$k 65536
  N=800 IMAGE=polystrata:tower $B live-512k-$k 524288 $Z
  N=800 IMAGE=$T $B short-512k-$k 524288 $Z
  N=800 IMAGE=$T $B full-512k-$k 524288 $Z STRATA_INK_ZHA_SHORT=0
done
N=1500 IMAGE=$T $B short-512k-1500 524288 $Z
N=1500 IMAGE=$T $B full-512k-1500 524288 $Z STRATA_INK_ZHA_SHORT=0
N=1500 IMAGE=polystrata:tower $B live-512k-1500 524288 $Z
