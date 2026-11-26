#!/bin/sh
# p24/chain6.sh IMAGE: short-chat probes (plain 64K, ZHA 1M) and the 128K grow/shrink/restore run on IMAGE.
D=/mnt/user/appdata/polystrata; export IMAGE=$1
trap 'touch $D/p24/chain6.done' EXIT
$D/p24/probe.sh plain-pool strata-inkling2-mini.json
$D/p24/probe.sh zha-pool p23-ink-zha-1m.json
TASKS=trim-tasks sh $D/p23/run.sh trim-pool p23-ink-zha-1m.json
