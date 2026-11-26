#!/bin/sh
# p26/chain11.sh: ctx-ds15 through the server with the drafter (strata-inkling2-mini-ds.json: live's config and
# --draft-model): probe.sh's short chats, then Pass 24's 128K needle run.
D=/mnt/user/appdata/polystrata; N1=polystrata:ctx-ds15
trap 'docker start PolyStrata >/dev/null; touch $D/p26/chain11.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
IMAGE=$N1 sh $D/p25/probe.sh ds15 strata-inkling2-mini-ds.json
docker stop PolyStrata >/dev/null 2>&1
IMAGE=$N1 TASKS=trim-tasks sh $D/p24/runm.sh inkling p26-trim-ds15 strata-inkling2-mini-ds.json
