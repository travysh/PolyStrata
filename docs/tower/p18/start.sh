#!/bin/sh
# start.sh CONFIG [ENV=VAL ...]: the next image as the test container, waits until loaded
C=$1; shift; E=""; for kv in "$@"; do E="$E -e $kv"; done
docker rm -f polystrata-next >/dev/null 2>&1
docker run -d --name polystrata-next --gpus all -p 5003:8080 --ulimit memlock=-1 --stop-timeout 60 --restart no \
  -v /mnt/cache/llm:/gguf -v /mnt/user/appdata/polystrata/data:/data -v /mnt/cache/appdata/polystrata/park:/park \
  -e TMPDIR=/park -e TZ=America/Los_Angeles $E polystrata:next-c868922e \
  .venv/bin/python serve/server.py --engine strata --config /data/config/$C --port 8080 --gpu 0 >/dev/null
docker cp /tmp/olddocs polystrata-next:/opt/strata/docs; docker cp /tmp/oldREADME.md polystrata-next:/opt/strata/README.md
until curl -s -m 5 localhost:5003/health | grep -q "\"loaded\": true"; do sleep 5; done
