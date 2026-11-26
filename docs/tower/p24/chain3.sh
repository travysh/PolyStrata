#!/bin/sh
# p24/chain3.sh IMAGE: short chats in server mode, twice each, on Inkling's 64K config, ZHA 1M with its leaves
# pinned as the context grows, and ZHA 1M pinned at the start (STRATA_INK_ZHA_PIN_ALL=1); then the grow, give back and
# restore run (trim-tasks: a 128K question, two short chats, a second question on the 128K text).
D=/mnt/user/appdata/polystrata; R=$D/p23/run.sh; export IMAGE=$1
trap 'touch $D/p24/chain3.done' EXIT
for r in 1 2; do
  TASKS=short-tasks sh $R s-plain-$r strata-inkling2-mini.json
  TASKS=short-tasks sh $R s-lazy-$r p23-ink-zha-1m.json
  TASKS=short-tasks sh $R s-pinall-$r p24-ink-zha-1m-pinall.json
done
TASKS=trim-tasks sh $R trim-lazy p23-ink-zha-1m.json
