#!/bin/sh
# p24/chain11.sh: the same picture tasks on the live image (polystrata:tower = 9172f75e), Inkling 64K.
D=/mnt/user/appdata/polystrata
trap "docker start PolyStrata >/dev/null; touch $D/p24/chain11.done" EXIT
IMAGE=polystrata:tower TASKS=img-tasks sh $D/p24/runm.sh inkling img-64k-live strata-inkling2-mini.json $(ls $D/p22/img-tasks | grep -- -off)
