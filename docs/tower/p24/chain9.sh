#!/bin/sh
D=/mnt/user/appdata/polystrata; export IMAGE=polystrata:ctx-513eceb1
trap 'touch $D/p24/chain9.done' EXIT
$D/p24/probe2.sh plain strata-inkling2-mini.json
$D/p24/probe2.sh zha512 p23-ink-zha-512k.json
