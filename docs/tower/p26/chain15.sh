#!/bin/sh
# p26/chain15.sh: the bench against server gap.  The six chats' prompts as the server encodes them (p26/chatN.txt)
# through strata-poly alone at temperature 0, 400 tokens (bench.sh's args; then with the server's
# --conversation-cache-mib auto too), and the same chats through the server (live's config), ctx-ds16, 512K, ZHA.
D=/mnt/user/appdata/polystrata; O=$D/p26; Z=STRATA_INK_ZHA=/opt/strata/terminal/data/zha/inkling-2-mini-refs-c64-r4.bin
I=polystrata:ctx-ds16
trap 'docker rm -f polystrata-next >/dev/null 2>&1; docker start PolyStrata >/dev/null; touch $O/chain15.done' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
B=$O/bench.sh
N=200 IMAGE=$I $B c15-warm 524288 $Z
for r in a b; do for c in 0 1 2 3 4 5; do
  N=400 IDS=chat$c.txt IMAGE=$I $B c15-cli-$r-chat$c 524288 $Z
done; done
for c in 0 1 2 3 4 5; do
  N=400 IDS=chat$c.txt ARGS="--conversation-cache-mib auto" IMAGE=$I $B c15-cc-chat$c 524288 $Z
done
docker run -d --name polystrata-next --gpus all -p 5003:8080 --ulimit memlock=-1 --restart no \
  -v /mnt/cache/llm:/gguf -v $D/data:/data -v $O:/p26 -v /mnt/cache/appdata/polystrata/park:/park \
  -e TMPDIR=/park -e STRATA_POLY_MIRROR=0 -e STRATA_CPU_NO_X4=1 $I \
  .venv/bin/python serve/server.py --engine strata --config /data/config/strata-inkling2-mini.json --port 8080 --gpu 0 >/dev/null
until curl -s -m 5 localhost:5003/health | grep -q "\"loaded\": true"; do sleep 5; done
docker exec polystrata-next sh -c "cd /opt/strata/terminal && .venv/bin/python /p26/chats.py 0 1 >/dev/null && .venv/bin/python /p26/chats.py 0 2" > $O/c15-server.log 2>&1
docker exec polystrata-next sh -c "grep 'strata serve:' /opt/strata/terminal/strata-inkling-ud-q4_k_xl.log" >> $O/c15-server.log 2>&1
docker exec polystrata-next sh -c "ps -eo pid,args | grep -E 'strata-poly|strata-vision' | grep -v grep" >> $O/c15-server.log 2>&1
