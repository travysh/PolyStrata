#!/bin/sh
# p23/chain4.sh: after chain3, build the corrected leaf-parking fix (ctx3/) and run the A B A A B swap test on it.
R=/mnt/user/appdata/polystrata/p22/runs; P=/mnt/user/appdata/polystrata/p23; C=/mnt/user/appdata/polystrata/ctx3
until [ -f $R/chain3.done ]; do sleep 30; done
T=$(cat $C/TOWER_COMMIT)
(cd $C && docker build -t polystrata:ctx-$T . > build.log 2>&1; echo $? > build.rc)
[ "$(cat $C/build.rc)" = 0 ] && IMAGE=polystrata:ctx-$T TASKS=swap-tasks sh $P/run.sh swap-fix2 p23-ink-zha-1m.json
touch $R/chain4.done
