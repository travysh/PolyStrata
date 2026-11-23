#!/bin/sh
# old.sh LABEL FATTN: the 0.1.45 image (tower-893468a5) with the template env, server patched to log PP lines to
# /data/p19-pp-LABEL.log, then cold reads at 32K and 64K (aba.py) to p19/LABEL.txt
L=$1; FA=$2
docker rm -f polystrata-next polystrata-old >/dev/null 2>&1
docker run -d --name polystrata-old --gpus all -p 5003:8080 --ulimit memlock=-1 --stop-timeout 60 --restart no \
  -v /mnt/user/appdata/polystrata/data:/data -v /mnt/cache/appdata/polystrata/park:/park -v /mnt/cache/llm:/gguf \
  -e FAMILY=cybertiel -e MODEL=UD-Q4_K_XL -e CONTEXT=262144 -e KV=int8 -e VISION=cpu -e STRATA_FATTN=$FA -e STRATA_FDEC=1 \
  -e STRATA_CPU_DOWN_ORDER=1 -e GGUF_DIR=/gguf -e GPU=0 -e TZ=America/Los_Angeles polystrata:tower-893468a5 >/dev/null
docker cp /mnt/user/appdata/polystrata/p19/pplog.py polystrata-old:/tmp/pplog.py
docker exec polystrata-old /opt/strata/.venv/bin/python /tmp/pplog.py /opt/strata/serve/server.py /data/p19-pp-$L.log
docker restart polystrata-old >/dev/null
until curl -s -m 5 localhost:5003/health | grep -q "\"loaded\": true"; do sleep 5; done
docker exec polystrata-old sh -c "cd /opt/strata && .venv/bin/python /data/decode.py warm 2 >/dev/null"
rm -f /mnt/user/appdata/polystrata/data/p19-pp-$L.log
docker exec polystrata-old sh -c "cd /opt/strata && .venv/bin/python /data/aba.py $L 24000 48000" > /mnt/user/appdata/polystrata/p19/$L.txt 2>&1
docker exec polystrata-old sh -c "cat /opt/strata/*.log" > /mnt/user/appdata/polystrata/p19/$L.log 2>&1
