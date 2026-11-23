#!/bin/sh
# nx.sh LABEL IMAGE CONFIG [ENV=VAL ...]: a 0.2.19 image as the test container (mirror off), server patched to log PP
# lines to /data/p19-pp-LABEL.log, warm-up, cold reads at 32K and 64K (aba.py) to p19/LABEL.txt, engine log to LABEL.log
L=$1; I=$2; C=$3; shift 3
E=""; for kv in "$@"; do E="$E -e $kv"; done
docker rm -f polystrata-next polystrata-old >/dev/null 2>&1
docker run -d --name polystrata-next --gpus all -p 5003:8080 --ulimit memlock=-1 --stop-timeout 60 --restart no \
  -v /mnt/cache/llm:/gguf -v /mnt/user/appdata/polystrata/data:/data -v /mnt/cache/appdata/polystrata/park:/park \
  -e TMPDIR=/park -e TZ=America/Los_Angeles -e STRATA_POLY_MIRROR=0 $E $I \
  .venv/bin/python serve/server.py --engine strata --config /data/config/$C --port 8080 --gpu 0 >/dev/null
docker cp /tmp/olddocs polystrata-next:/opt/strata/docs; docker cp /tmp/oldREADME.md polystrata-next:/opt/strata/README.md
docker cp /mnt/user/appdata/polystrata/p19/pplog.py polystrata-next:/tmp/pplog.py
docker exec polystrata-next /opt/strata/terminal/.venv/bin/python /tmp/pplog.py /opt/strata/terminal/serve/server.py /data/p19-pp-$L.log
docker restart polystrata-next >/dev/null
until curl -s -m 5 localhost:5003/health | grep -q "\"loaded\": true"; do sleep 5; done
docker exec polystrata-next sh -c "cd /opt/strata/terminal && .venv/bin/python /data/decode.py warm 2 >/dev/null"
rm -f /mnt/user/appdata/polystrata/data/p19-pp-$L.log
docker exec polystrata-next sh -c "cd /opt/strata/terminal && PYTHONPATH=/opt/strata/terminal .venv/bin/python /data/aba.py $L 24000 48000" > /mnt/user/appdata/polystrata/p19/$L.txt 2>&1
docker exec polystrata-next sh -c "cat /opt/strata/terminal/*.log" > /mnt/user/appdata/polystrata/p19/$L.log 2>&1
