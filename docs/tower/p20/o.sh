#!/bin/sh
# o.sh LABEL: the 0.1.45 image (tower-893468a5) with its live template environment (pre-0219), the same set as t.sh plus
# dprobe.py. Results: p20/LABEL.txt and data/upd-LABEL.txt.
L=$1; D=/mnt/user/appdata/polystrata; O=$D/p20/$L.txt
docker rm -f polystrata-next >/dev/null 2>&1
docker run -d --name polystrata-next --gpus all -p 5003:8080 --ulimit memlock=-1 --stop-timeout 60 --restart no \
  -v $D/data:/data -v /mnt/cache/appdata/polystrata/park:/park -v /mnt/cache/llm:/gguf \
  -e FAMILY=cybertiel -e MODEL=UD-Q4_K_XL -e CONTEXT=262144 -e KV=int8 -e VISION=cpu -e STRATA_FATTN=1 -e STRATA_FDEC=1 \
  -e STRATA_CPU_DOWN_ORDER=1 -e GGUF_DIR=/gguf -e GPU=0 -e TZ=America/Los_Angeles polystrata:tower-893468a5 >/dev/null
until curl -s -m 5 localhost:5003/health | grep -q "\"loaded\": true"; do sleep 5; done
P="cd /opt/strata && PYTHONPATH=/opt/strata"
{
echo "== $L polystrata:tower-893468a5 $(date)"
docker exec polystrata-next sh -c "$P .venv/bin/python /data/decode.py warm 2 >/dev/null"
docker exec polystrata-next sh -c "$P .venv/bin/python /data/features.py"
docker exec polystrata-next sh -c "$P .venv/bin/python -m unittest serve.test_grammar serve.test_logit_bias serve.test_prefill 2>&1 | tail -3"
docker exec polystrata-next sh -c "$P sh /data/runbench.sh $L .venv/bin/python"
docker exec polystrata-next sh -c "$P .venv/bin/python /data/ldecode.py $L-ldec 8000 24000 48000"
docker exec polystrata-next sh -c "$P .venv/bin/python /data/dprobe.py $L-dp"
docker exec polystrata-next sh -c "grep -hE \"STRATA_FATTN|STRATA_FDEC|STRATA_CPU_DOWN|error|Error\" /opt/strata/*.log | sort | uniq -c | head -20"
echo "== done $(date)"
} > $O 2>&1
