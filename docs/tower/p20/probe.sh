#!/bin/sh
# probe.sh LABEL IMAGE [ENV=VAL ...]: dprobe.py in the test container with live environment, to p20/probe-LABEL.txt
L=$1; I=$2; shift 2; E=""; for kv in "$@"; do E="$E -e $kv"; done; D=/mnt/user/appdata/polystrata
docker rm -f polystrata-next >/dev/null 2>&1
docker run -d --name polystrata-next --gpus all -p 5003:8080 --ulimit memlock=-1 --stop-timeout 60 --restart no \
  -v /mnt/cache/llm:/gguf -v $D/data:/data -v /mnt/cache/appdata/polystrata/park:/park \
  -e TMPDIR=/park -e STRATA_POLY_MIRROR=0 -e TZ=America/Los_Angeles $E $I \
  .venv/bin/python serve/server.py --engine strata --config /data/config/strata-cybertiel-next.json --port 8080 --gpu 0 >/dev/null
docker cp /tmp/olddocs polystrata-next:/opt/strata/docs; docker cp /tmp/oldREADME.md polystrata-next:/opt/strata/README.md
until curl -s -m 5 localhost:5003/health | grep -q "\"loaded\": true"; do sleep 5; done
P="cd /opt/strata/terminal && PYTHONPATH=/opt/strata/terminal"
docker exec polystrata-next sh -c "$P .venv/bin/python /data/decode.py warm 2 >/dev/null"
docker exec polystrata-next sh -c "$P .venv/bin/python /data/dprobe.py $L" > $D/p20/probe-$L.txt 2>&1
