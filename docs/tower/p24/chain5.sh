#!/bin/sh
D=/mnt/user/appdata/polystrata
IMAGE=polystrata:ctx-97665a39 TASKS=trim2-tasks sh $D/p23/run.sh trim2-lazy p23-ink-zha-1m.json
touch $D/p24/chain5.done
