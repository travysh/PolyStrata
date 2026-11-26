#!/bin/sh
trap "docker start PolyStrata >/dev/null; touch /mnt/user/appdata/polystrata/p24/chain1.done" EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=/mnt/user/appdata/polystrata/p24/bench.sh
$B plain-64k 65536
$B zha-1m 1048576 STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
$B zha-64k 65536 STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
$B zha-1m-pool0 1048576 STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin STRATA_INK_ZHA_POOL=0
$B plain-64k-b 65536
$B zha-1m-b 1048576 STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
