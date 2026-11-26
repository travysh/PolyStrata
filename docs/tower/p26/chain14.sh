#!/bin/sh
# p26/chain14.sh: why sampled chats lose with the drafter.  ctx-ds16 through the server with live's config (plain)
# and with the drafter config (ds), in turns: six chats at temperature 0, then at 0.6, two rounds each; the engine
# log and the container log kept (p26/c14-*.log).
D=/mnt/user/appdata/polystrata; O=$D/p26
trap 'docker rm -f polystrata-next >/dev/null 2>&1; docker start PolyStrata >/dev/null; touch $O/chain14.done' EXIT
docker stop PolyStrata >/dev/null 2>&1
for r in "plain strata-inkling2-mini.json a" "ds strata-inkling2-mini-ds.json a" "ds strata-inkling2-mini-ds.json b" "plain strata-inkling2-mini.json b"; do
  set -- $r
  docker rm -f polystrata-next >/dev/null 2>&1
  docker run -d --name polystrata-next --gpus all -p 5003:8080 --ulimit memlock=-1 --restart no \
    -v /mnt/cache/llm:/gguf -v $D/data:/data -v $O:/p26 -v /mnt/cache/appdata/polystrata/park:/park \
    -e TMPDIR=/park -e STRATA_POLY_MIRROR=0 -e STRATA_CPU_NO_X4=1 polystrata:ctx-ds16 \
    .venv/bin/python serve/server.py --engine strata --config /data/config/$2 --port 8080 --gpu 0 >/dev/null
  until curl -s -m 5 localhost:5003/health | grep -q "\"loaded\": true"; do sleep 5; done
  L=$O/c14-$1-$3.log
  docker exec polystrata-next sh -c "cd /opt/strata/terminal && .venv/bin/python /p26/chats.py 0 1 >/dev/null && .venv/bin/python /p26/chats.py 0 2 && .venv/bin/python /p26/chats.py 0.6 2" > $L 2>&1
  docker exec polystrata-next sh -c "grep 'strata serve:' /opt/strata/terminal/strata-inkling-ud-q4_k_xl.log" >> $L 2>&1
  docker logs polystrata-next > $O/c14-$1-$3.server.log 2>&1
done
