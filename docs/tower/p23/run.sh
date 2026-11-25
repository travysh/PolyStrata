#!/bin/sh
# p23/run.sh LABEL CONFIG IDS...: one Pass 23 run on Tower. Starts Inkling with /data/config/CONFIG in the test
# container (image $IMAGE, default polystrata:ctx-a583121c) through p22/model.sh, sends IDS from p22/$TASKS (default ctx-tasks), keeps the
# engine log and the card's memory every 10 s, and starts live PolyStrata again at the end.
L=$1; C=$2; shift 2; D=/mnt/user/appdata/polystrata; O=$D/p22/runs
trap 'kill $W 2>/dev/null; docker start PolyStrata >/dev/null; touch $O/$L.done' EXIT
( while sleep 10; do echo "$(date +%T) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) $(free -m | awk '/Mem/{print "ram used " $3 " avail " $7}')"; done > $O/$L.mem ) & W=$!
( sleep 60; until [ -f $O/$L.done ]; do docker cp polystrata-next:/opt/strata/terminal/strata-inkling-ud-q4_k_xl.log $O/$L.engine.log 2>/dev/null; sleep 10; done ) &
IMAGE=${IMAGE:-polystrata:ctx-a583121c} CONFIG=/data/config/$C TASKS=${TASKS:-ctx-tasks} sh $D/p22/model.sh inkling $L 1 "$@"
