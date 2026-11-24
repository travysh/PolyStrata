#!/bin/sh
# p21/chain2.sh IMAGE: the Inkling set on IMAGE (fix.txt), then Cyber-Tiel's live config with features.py and the
# image's unit tests (fix-ct.txt), then live back
I=$1; D=/mnt/user/appdata/polystrata; cd $D/p21
./t.sh fix $I
docker rm -f polystrata-next >/dev/null 2>&1
docker run -d --name polystrata-next --gpus all -p 5003:8080 --ulimit memlock=-1 --stop-timeout 60 --restart no \
  -v /mnt/cache/llm:/gguf -v $D/data:/data -v /mnt/cache/appdata/polystrata/park:/park \
  -e TMPDIR=/park -e STRATA_POLY_MIRROR=0 -e STRATA_CPU_NO_X4=1 -e TZ=America/Los_Angeles $I \
  .venv/bin/python serve/server.py --engine strata --config /data/config/strata-cybertiel-next.json --port 8080 --gpu 0 >/dev/null
until curl -s -m 5 localhost:5003/health | grep -q "\"loaded\": true"; do sleep 5; done
P="cd /opt/strata/terminal && PYTHONPATH=/opt/strata/terminal"
{ docker exec polystrata-next sh -c "$P .venv/bin/python /data/features.py"
  docker exec polystrata-next sh -c "$P .venv/bin/python -m unittest serve.test_grammar serve.test_logit_bias serve.test_prefill serve.test_inkling 2>&1 | tail -3"
} > fix-ct.txt 2>&1
docker rm -f polystrata-next >/dev/null; docker start PolyStrata >/dev/null
date > chain2.done
