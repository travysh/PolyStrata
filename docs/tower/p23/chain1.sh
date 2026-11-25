#!/bin/sh
# p23/chain1.sh: after the ZHA ladder, the full-attention control at 128K and the prompt-batch variants (128K level).
R=/mnt/user/appdata/polystrata/p22/runs; P=/mnt/user/appdata/polystrata/p23
until [ -f $R/zha1m-ladder.done ]; do sleep 30; done
sh $P/run.sh q8-128k-ctl p23-ink-q8-128k.json long5-q1 long5-q2 long5-q3 long5-q4 long5-q5 long5-q6
sh $P/run.sh zha-pb2048 p23-ink-zha-1m-pb2048.json long5-q1 long5-q2
sh $P/run.sh zha-pb4096 p23-ink-zha-1m-pb4096.json long5-q1 long5-q2
touch $R/chain1.done
