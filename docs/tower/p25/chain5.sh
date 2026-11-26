#!/bin/sh
# p25/chain5.sh: tower-ctx fd6d84bc (fewer launches) against e3331820 (ring-only step) and the live image, engine by
# itself on the short prompt (800 tokens) and the 4,521-token prompt (400), interleaved; 1,500 tokens on fd6d84bc;
# probe.sh's short chats on it and on live; then Pass 24's 128K needle run (p22/trim-tasks: a 128K question, two short
# chats, a second 128K question) on it with live's Inkling config.
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
N1=polystrata:ctx-fd6d84bc; E=polystrata:ctx-e3331820; L=polystrata:tower
trap 'docker start PolyStrata >/dev/null; touch $D/p25/chain5.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p25/bench.sh
for r in "$L c5-live" "$E c5-e333" "$N1 c5-fd6d" "$N1 c5-fd6d-b" "$E c5-e333-b" "$L c5-live-b"; do
  set -- $r
  N=800 IMAGE=$1 $B $2-short 524288 $Z
  IDS=ids-4k.txt IMAGE=$1 $B $2-mid 524288 $Z
done
N=1500 IMAGE=$N1 $B c5-fd6d-1500 524288 $Z
IMAGE=$N1 sh $D/p25/probe.sh fd6d strata-inkling2-mini.json
IMAGE=$L sh $D/p25/probe.sh live-c5 strata-inkling2-mini.json
docker stop PolyStrata >/dev/null 2>&1
IMAGE=$N1 TASKS=trim-tasks sh $D/p24/runm.sh inkling p25-trim-fd6d strata-inkling2-mini.json
