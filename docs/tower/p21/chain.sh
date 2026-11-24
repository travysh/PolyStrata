#!/bin/sh
# p21/chain.sh: round b (setup's config again), then c (acc.json: without the three small-card settings)
cd /mnt/user/appdata/polystrata/p21
./t.sh b polystrata:tower-9172f75e
CFG=/data/p21/acc.json ./t.sh c polystrata:tower-9172f75e
touch chain.done
