#!/bin/sh
# p26/chain12.sh: why the drafter config's server refused the 128K needle question (HTTP 400): ctx-ds15 with
# strata-inkling2-mini-ds.json, the question alone, then the engine log.
D=/mnt/user/appdata/polystrata; O=$D/p26/c12.out
trap 'docker rm -f polystrata-next >/dev/null 2>&1; docker start PolyStrata >/dev/null; touch $D/p26/chain12.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
docker run -d --name polystrata-next --gpus all -p 5003:8080 --ulimit memlock=-1 --restart no \
  -v /mnt/cache/llm:/gguf -v $D/data:/data -v $D/p22:/p22 -v $D/p26:/p26 -v /mnt/cache/appdata/polystrata/park:/park \
  -e TMPDIR=/park -e STRATA_POLY_MIRROR=0 -e STRATA_CPU_NO_X4=1 polystrata:ctx-ds15 \
  .venv/bin/python serve/server.py --engine strata --config /data/config/strata-inkling2-mini-ds.json --port 8080 --gpu 0 >/dev/null
until curl -s -m 5 localhost:5003/health | grep -q "\"loaded\": true"; do sleep 5; done
docker exec polystrata-next sh -c "cd /opt/strata/terminal && .venv/bin/python /p26/why400.py /p22/trim-tasks/t1-long5-q1/task.json" > $O 2>&1
docker logs polystrata-next 2>&1 | tail -40 >> $O
docker exec polystrata-next sh -c "tail -40 /opt/strata/terminal/strata-inkling-ud-q4_k_xl.log" >> $O 2>&1
