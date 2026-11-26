#!/bin/sh
# p24/probe.sh LABEL CONFIG: start CONFIG in the test container (image $IMAGE), send three short chats, ask the
# engine for its VRAM figures (POST /v1/vram {}) after each, and keep the engine log.  Live PolyStrata restarts at the end.
D=/mnt/user/appdata/polystrata; L=$1; C=/data/config/$2; O=$D/p24/$L.probe
trap 'docker cp polystrata-next:/opt/strata/terminal/strata-inkling-ud-q4_k_xl.log $O.engine.log 2>/dev/null; docker rm -f polystrata-next >/dev/null; docker start PolyStrata >/dev/null' EXIT
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
docker run -d --name polystrata-next --gpus all -p 5003:8080 --ulimit memlock=-1 --restart no \
  -v /mnt/cache/llm:/gguf -v $D/data:/data -v /mnt/cache/appdata/polystrata/park:/park \
  -e TMPDIR=/park -e STRATA_POLY_MIRROR=0 -e STRATA_CPU_NO_X4=1 $IMAGE \
  .venv/bin/python serve/server.py --engine strata --config $C --port 8080 --gpu 0 >/dev/null
until curl -s -m 5 localhost:5003/health | grep -q '"loaded": true'; do sleep 5; done
echo "== $L $IMAGE $C" > $O
for q in "Tell me a story about a fox." "Explain how a refrigerator works." "Write a poem about rain."; do
  curl -s -m 600 localhost:5003/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":400,\"temperature\":0}" | grep -o '"completion_tokens": [0-9]*' >> $O
  echo "vram: $(curl -s -m 120 -X POST localhost:5003/v1/vram -H 'Content-Type: application/json' -d '{}')" >> $O
  echo "smi: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $O
done
