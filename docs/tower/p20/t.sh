#!/bin/sh
# t.sh LABEL IMAGE: the 0.2.20 update's comparison set in the test container with live's environment.
# Results: p20/LABEL.txt (features, unit tests, long decode, engine's kernel lines) and data/upd-LABEL.txt (runbench).
L=$1; I=$2; shift 2; E=""; for kv in "$@"; do E="$E -e $kv"; done; D=/mnt/user/appdata/polystrata; O=$D/p20/$L.txt
docker rm -f polystrata-next >/dev/null 2>&1
docker run -d --name polystrata-next --gpus all -p 5003:8080 --ulimit memlock=-1 --stop-timeout 60 --restart no \
  -v /mnt/cache/llm:/gguf -v $D/data:/data -v /mnt/cache/appdata/polystrata/park:/park \
  -e TMPDIR=/park -e STRATA_POLY_MIRROR=0 -e TZ=America/Los_Angeles $E $I \
  .venv/bin/python serve/server.py --engine strata --config /data/config/strata-cybertiel-next.json --port 8080 --gpu 0 >/dev/null
docker cp /tmp/olddocs polystrata-next:/opt/strata/docs; docker cp /tmp/oldREADME.md polystrata-next:/opt/strata/README.md
until curl -s -m 5 localhost:5003/health | grep -q "\"loaded\": true"; do sleep 5; done
P="cd /opt/strata/terminal && PYTHONPATH=/opt/strata/terminal"
{
echo "== $L $I $(date)"
docker exec polystrata-next sh -c "$P .venv/bin/python /data/decode.py warm 2 >/dev/null"
docker exec polystrata-next sh -c "$P .venv/bin/python /data/features.py"
docker exec polystrata-next sh -c "$P .venv/bin/python -m unittest serve.test_grammar serve.test_logit_bias serve.test_prefill 2>&1 | tail -3"
docker exec polystrata-next sh -c "$P sh /data/runbench.sh $L .venv/bin/python"
docker exec polystrata-next sh -c "$P .venv/bin/python /data/ldecode.py $L-ldec 8000 24000 48000"
docker exec polystrata-next sh -c "$P .venv/bin/python /data/dprobe.py $L-dp"
docker exec polystrata-next sh -c "grep -hE 'STRATA_FATTN|STRATA_FDEC|error|Error' /opt/strata/terminal/*.log | sort | uniq -c | head -20"
echo "== done $(date)"
} > $O 2>&1
