#!/bin/sh
# p24/chain2.sh: ZHA at 256K, 512K and 1M and plain 64K, 800 tokens each, with vmstat (page-ins) and free after.
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
trap 'kill $V 2>/dev/null; docker start PolyStrata >/dev/null; touch $D/p24/chain2.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p24/bench.sh
for r in "zha-256k 262144" "zha-512k 524288" "zha-1m-c 1048576" "plain-64k-c 65536"; do
  set -- $r
  ( vmstat 2 > $D/p24/$1.vmstat ) & V=$!
  if [ $1 = plain-64k-c ]; then N=800 $B $1 $2; else N=800 $B $1 $2 $Z; fi
  kill $V; free -m > $D/p24/$1.free
done
