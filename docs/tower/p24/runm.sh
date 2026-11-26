#!/bin/sh
# p24/runm.sh MODEL LABEL CONFIG [IDS]: p23/run.sh for either model (cybertiel or inkling): CONFIG from
# /data/config, tasks p22/$TASKS, image $IMAGE; engine log and memory kept in p22/runs.  Does not start live again.
M=$1; L=$2; C=$3; shift 3; D=/mnt/user/appdata/polystrata; O=$D/p22/runs
case $M in cybertiel) G=strata-cybertiel-next.log ;; *) G=strata-inkling-ud-q4_k_xl.log ;; esac
( while sleep 10; do echo "$(date +%T) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) $(free -m | awk '/Mem/{print "ram used " $3 " avail " $7}')"; done > $O/$L.mem ) & W=$!
( sleep 60; until [ -f $O/$L.done ]; do docker cp polystrata-next:/opt/strata/terminal/$G $O/$L.engine.log 2>/dev/null; sleep 10; done ) &
CONFIG=/data/config/$C sh $D/p22/model.sh $M $L 1 "$@"
kill $W; touch $O/$L.done
