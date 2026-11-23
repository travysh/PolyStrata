#!/bin/sh
# v.sh LABEL CONFIG [ENV=VAL ...]: next image, cold reads at 32K and 64K (aba.py), engine log to p19/LABEL.log
L=$1; C=$2; shift 2
/mnt/user/appdata/polystrata/next/start.sh $C STRATA_POLY_MIRROR=0 "$@"
docker exec polystrata-next sh -c "cd /opt/strata/terminal && .venv/bin/python /data/decode.py warm 2 >/dev/null; PYTHONPATH=/opt/strata/terminal .venv/bin/python /data/aba.py $L 24000 48000" > /mnt/user/appdata/polystrata/p19/$L.txt 2>&1
docker exec polystrata-next sh -c "cat /opt/strata/terminal/*.log" > /mnt/user/appdata/polystrata/p19/$L.log 2>&1
