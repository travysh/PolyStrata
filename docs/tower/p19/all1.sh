#!/bin/sh
V=/mnt/user/appdata/polystrata/p19/v.sh
docker stop -t 60 PolyStrata
$V base strata-cybertiel-next.json
$V fattn0 strata-cybertiel-next.json STRATA_FATTN=0
$V every0 p19-every0.json
$V pc0 p19-pc0.json
docker rm -f polystrata-next
echo done > /mnt/user/appdata/polystrata/p19/all1.done
