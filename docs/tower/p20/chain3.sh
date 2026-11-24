#!/bin/sh
# build next-9172f75e, then 0.1.45 against it with STRATA_CPU_NO_X4=1, twice each in turn
cd /mnt/user/appdata/polystrata/next && docker build -t polystrata:next-9172f75e . > build.log 2>&1 || { echo build failed > /mnt/user/appdata/polystrata/p20/chain3.done; exit 1; }
cd /mnt/user/appdata/polystrata/p20
docker stop PolyStrata >/dev/null
./o.sh v145-c; ./t.sh x4-a polystrata:next-9172f75e STRATA_CPU_NO_X4=1
./o.sh v145-d; ./t.sh x4-b polystrata:next-9172f75e STRATA_CPU_NO_X4=1
docker rm -f polystrata-next >/dev/null
docker start PolyStrata >/dev/null
date > chain3.done
