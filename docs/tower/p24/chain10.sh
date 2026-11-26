#!/bin/sh
# p24/chain10.sh: the 35 picture tasks (thinking off) on Inkling 64K, image ctx-513eceb1, to compare with img-512k.
D=/mnt/user/appdata/polystrata
trap "docker start PolyStrata >/dev/null; touch $D/p24/chain10.done" EXIT
IMAGE=polystrata:ctx-513eceb1 TASKS=img-tasks sh $D/p24/runm.sh inkling img-64k strata-inkling2-mini.json $(ls $D/p22/img-tasks | grep -- -off)
