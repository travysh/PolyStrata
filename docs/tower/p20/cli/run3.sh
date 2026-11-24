#!/bin/sh
cd /mnt/user/appdata/polystrata/data/p20cli
docker stop PolyStrata >/dev/null
N=polystrata:next-aadae31e; V=polystrata:tower-c868922e
for r in 1 2; do
./cli.sh v219-short-$r $V short - STRATA_POLY_MIRROR=0
./cli.sh new-short-r3-$r $N short - STRATA_POLY_MIRROR=0
./cli.sh v219-long-$r $V long - STRATA_POLY_MIRROR=0
./cli.sh new-long-r3-$r $N long - STRATA_POLY_MIRROR=0
done
./cli.sh new-short-mirror $N short -
docker start PolyStrata >/dev/null
date > run3.done
