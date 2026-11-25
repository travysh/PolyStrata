#!/bin/sh
# p23/chain2.sh: after chain1, build the image with the leaf-parking fix (ctx2/), then the A B A A B swap test on
# the image before it and on the fixed one.
R=/mnt/user/appdata/polystrata/p22/runs; P=/mnt/user/appdata/polystrata/p23; C=/mnt/user/appdata/polystrata/ctx2
until [ -f $R/chain1.done ]; do sleep 30; done
T=$(cat $C/TOWER_COMMIT)
(cd $C && docker build -t polystrata:ctx-$T . > build.log 2>&1; echo $? > build.rc)
TASKS=swap-tasks sh $P/run.sh swap-old p23-ink-zha-1m.json
[ "$(cat $C/build.rc)" = 0 ] && IMAGE=polystrata:ctx-$T TASKS=swap-tasks sh $P/run.sh swap-fix p23-ink-zha-1m.json
touch $R/chain2.done
