#!/bin/sh
# four-row path on/off (p20-x4) against 0.2.19 and 0.1.45, engine alone; a warm-up run first, discarded
cd /mnt/user/appdata/polystrata/data/p20cli
docker stop PolyStrata >/dev/null
X=polystrata:p20-x4; V=polystrata:tower-c868922e; O=polystrata:tower-893468a5
./cli.sh warm $X short - STRATA_POLY_MIRROR=0
for r in 1 2; do
./cli.sh x4on-short-$r $X short - STRATA_POLY_MIRROR=0
./cli.sh x4off-short-$r $X short - STRATA_POLY_MIRROR=0 STRATA_CPU_NO_X4=1
./cli.sh v219-short-r4-$r $V short - STRATA_POLY_MIRROR=0
./cli.sh old-short-r4-$r $O short - STRATA_FATTN=1 STRATA_FDEC=1 STRATA_CPU_DOWN_ORDER=1
./cli.sh x4on-long-$r $X long - STRATA_POLY_MIRROR=0
./cli.sh x4off-long-$r $X long - STRATA_POLY_MIRROR=0 STRATA_CPU_NO_X4=1
./cli.sh v219-long-r4-$r $V long - STRATA_POLY_MIRROR=0
./cli.sh old-long-r4-$r $O long - STRATA_FATTN=1 STRATA_FDEC=1 STRATA_CPU_DOWN_ORDER=1
done
docker start PolyStrata >/dev/null
date > run4.done
