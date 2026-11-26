#!/bin/sh
# p24/probe2.sh LABEL CONFIG: fresh prompts of about 40, 300, 2,000 and 8,000 tokens (each new text, nothing reused),
# 32 tokens written each, twice; the engine log is kept for the read times.  Image $IMAGE.  Live restarts at the end.
D=/mnt/user/appdata/polystrata; L=$1; C=/data/config/$2; O=$D/p24/$L.probe2
trap 'docker cp polystrata-next:/opt/strata/terminal/strata-inkling-ud-q4_k_xl.log $O.engine.log 2>/dev/null; docker rm -f polystrata-next >/dev/null; docker start PolyStrata >/dev/null' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
docker run -d --name polystrata-next --gpus all -p 5003:8080 --ulimit memlock=-1 --restart no \
  -v /mnt/cache/llm:/gguf -v $D/data:/data -v /mnt/cache/appdata/polystrata/park:/park \
  -e TMPDIR=/park -e STRATA_POLY_MIRROR=0 -e STRATA_CPU_NO_X4=1 $IMAGE \
  .venv/bin/python serve/server.py --engine strata --config $C --port 8080 --gpu 0 >/dev/null
until curl -s -m 5 localhost:5003/health | grep -q '"loaded": true'; do sleep 5; done
echo "== $L $IMAGE $C" > $O
for round in a b c; do
  for words in 25 220 1500 6000; do
    text=$(awk -v n=$words -v s="$round$words" 'BEGIN{srand(length(s)*7+n); for(i=0;i<n;i++) printf "w%d ", int(rand()*100000); }')
    curl -s -m 600 localhost:5003/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Round $round. Say how many words follow, roughly: $text\"}],\"max_tokens\":32,\"temperature\":0,\"reasoning_effort\":\"none\"}" \
      | grep -o '"prompt_tokens": [0-9]*' >> $O
  done
done
