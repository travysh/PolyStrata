#!/bin/sh
export IMAGE=polystrata:ctx-97665a39
/mnt/user/appdata/polystrata/p24/probe.sh plain strata-inkling2-mini.json
/mnt/user/appdata/polystrata/p24/probe.sh lazy p23-ink-zha-1m.json
touch /mnt/user/appdata/polystrata/p24/chain4.done
