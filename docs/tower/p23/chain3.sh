#!/bin/sh
# p23/chain3.sh: after chain2, short prompts (thinking off, 512 tokens) on the fixed image with live's Inkling config
# and with the ZHA configs, for decode and short-prompt speed.
R=/mnt/user/appdata/polystrata/p22/runs; P=/mnt/user/appdata/polystrata/p23
until [ -f $R/chain2.done ]; do sleep 30; done
I=polystrata:ctx-$(cat /mnt/user/appdata/polystrata/ctx2/TOWER_COMMIT)
for c in strata-inkling2-mini p23-ink-zha-1m-pb2048 p23-ink-zha-512k-pb2048 p23-ink-zha-1m; do
  IMAGE=$I TASKS=short-tasks sh $P/run.sh short-$c $c.json
done
touch $R/chain3.done
