#!/bin/sh
cd /mnt/user/appdata/polystrata/data/p20cli
docker stop PolyStrata >/dev/null
N=polystrata:next-aadae31e; O=polystrata:tower-893468a5
for r in 1 2; do
./cli.sh new-short-$r $N short - STRATA_POLY_MIRROR=0
./cli.sh old-short-$r $O short - STRATA_FATTN=1 STRATA_FDEC=1 STRATA_CPU_DOWN_ORDER=1
./cli.sh new-long-$r $N long - STRATA_POLY_MIRROR=0
./cli.sh old-long-$r $O long - STRATA_FATTN=1 STRATA_FDEC=1 STRATA_CPU_DOWN_ORDER=1
done
cp new-long-1.out long.follow; cp new-short-1.out short.follow
./cli.sh new-long-f $N long long.follow STRATA_POLY_MIRROR=0
./cli.sh old-long-f $O long long.follow STRATA_FATTN=1 STRATA_FDEC=1 STRATA_CPU_DOWN_ORDER=1
./cli.sh old-long-f-cdo0 $O long long.follow STRATA_FATTN=1 STRATA_FDEC=1 STRATA_CPU_DOWN_ORDER=0
./cli.sh new-short-f $N short short.follow STRATA_POLY_MIRROR=0
./cli.sh old-short-f $O short short.follow STRATA_FATTN=1 STRATA_FDEC=1 STRATA_CPU_DOWN_ORDER=1
docker start PolyStrata >/dev/null
date > run1.done
