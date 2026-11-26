#!/bin/sh
# p25/chain2.sh: a 4,521-token prompt (p25/ids-4k.txt), 400 tokens, engine by itself: plain 64K and ZHA 512K on the
# live image, ZHA 512K on the test image; then probe.sh's three short chats through the server: plain 64K
# (p25-ink-64k.json = live's strata-inkling2-mini.json.pre-zha) and live's 512K config on the live image, live's
# 512K config on the test image.
D=/mnt/user/appdata/polystrata; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
T=${T:-polystrata:ctx-e3331820}
trap 'docker start PolyStrata >/dev/null; touch $D/p25/chain2.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$D/p25/bench.sh
IDS=ids-4k.txt IMAGE=polystrata:tower $B mid-plain-64k 65536
IDS=ids-4k.txt IMAGE=polystrata:tower $B mid-live-512k 524288 $Z
IDS=ids-4k.txt IMAGE=$T $B mid-short-512k 524288 $Z
IDS=ids-4k.txt IMAGE=polystrata:tower $B mid-plain-64k-b 65536
IDS=ids-4k.txt IMAGE=polystrata:tower $B mid-live-512k-b 524288 $Z
IDS=ids-4k.txt IMAGE=$T $B mid-short-512k-b 524288 $Z
IMAGE=polystrata:tower sh $D/p25/probe.sh plain p25-ink-64k.json
IMAGE=polystrata:tower sh $D/p25/probe.sh live strata-inkling2-mini.json
IMAGE=$T sh $D/p25/probe.sh short strata-inkling2-mini.json
