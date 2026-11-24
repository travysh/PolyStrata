#!/bin/sh
cd /mnt/user/appdata/polystrata/data/p20cli
docker stop PolyStrata >/dev/null
N=polystrata:next-aadae31e; O=polystrata:tower-893468a5
for r in 1 2; do
THREADS=12 ./cli.sh new-short-t12-$r $N short - STRATA_POLY_MIRROR=0
./cli.sh old-short-r2-$r $O short - STRATA_FATTN=1 STRATA_FDEC=1 STRATA_CPU_DOWN_ORDER=1
THREADS=12 ./cli.sh new-long-t12-$r $N long - STRATA_POLY_MIRROR=0
./cli.sh old-long-r2-$r $O long - STRATA_FATTN=1 STRATA_FDEC=1 STRATA_CPU_DOWN_ORDER=1
done
THREADS=10 ./cli.sh new-short-t10 $N short - STRATA_POLY_MIRROR=0
THREADS=10 ./cli.sh new-long-t10 $N long - STRATA_POLY_MIRROR=0
docker start PolyStrata >/dev/null
date > run2.done
