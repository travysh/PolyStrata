#!/bin/sh
docker stop PolyStrata >/dev/null
./probe.sh new polystrata:next-aadae31e
./probe.sh old polystrata:tower-c868922e
./probe.sh new-fdec0 polystrata:next-aadae31e STRATA_FDEC=0
docker rm -f polystrata-next >/dev/null
docker start PolyStrata >/dev/null
date > pchain.done
