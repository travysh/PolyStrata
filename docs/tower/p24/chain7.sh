#!/bin/sh
# p24/chain7.sh: short-chat probes of ZHA at 256K and 512K on image ctx-513eceb1.
D=/mnt/user/appdata/polystrata; export IMAGE=polystrata:ctx-513eceb1
trap "touch $D/p24/chain7.done" EXIT
$D/p24/probe.sh zha-256k-pool p23-ink-zha-256k.json
$D/p24/probe.sh zha-512k-pool p23-ink-zha-512k.json
