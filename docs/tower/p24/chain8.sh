#!/bin/sh
# p24/chain8.sh: before going live with Inkling at 512K on image ctx-513eceb1: its grow/restore run and the 35
# picture tasks with thinking off; Cyber-Tiel's live config on the live image and on ctx-513eceb1 (a 128K prompt, two
# short chats).  Live PolyStrata starts again at the end.
D=/mnt/user/appdata/polystrata; R=$D/p24/runm.sh; N=polystrata:ctx-513eceb1
trap 'docker start PolyStrata >/dev/null; touch $D/p24/chain8.done' EXIT
IMAGE=$N TASKS=trim-tasks sh $R inkling trim-512k p23-ink-zha-512k.json
IMAGE=$N TASKS=img-tasks sh $R inkling img-512k p23-ink-zha-512k.json $(ls $D/p22/img-tasks | grep -- -off)
IMAGE=polystrata:tower TASKS=trim-tasks sh $R cybertiel cyb-live strata-cybertiel-next.json long5-q1 sh1 sh2
IMAGE=$N TASKS=trim-tasks sh $R cybertiel cyb-ctx strata-cybertiel-next.json long5-q1 sh1 sh2
