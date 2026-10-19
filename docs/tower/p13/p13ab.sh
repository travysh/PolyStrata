#!/bin/sh
# Pass 13 Phase 3 A/B: per run "variant:dir" (base = polystrata:tower at 64f17cc2, b36 = polystrata:tower-b36 with
# PR 36), a container strata-p13 with the PolyStrata template's settings, then 3 short requests, the 107K recall
# warm-up, multiturn.py dir $TURNS.  PolyStrata must be stopped; it is started again at the end.
D=/mnt/user/appdata/polystrata/data; F=/mnt/cache/llm/Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf; C=strata-p13
TURNS=${TURNS:-5}
mk() {
  docker rm -f $C >/dev/null 2>&1
  docker create --name $C --gpus all -p 8080:8080 --ulimit memlock=-1 --stop-timeout 60 --restart no \
    -v $D:/data -v /mnt/cache/llm:/gguf -e FAMILY=cybertiel -e MODEL=UD-Q4_K_XL -e GGUF_DIR=/gguf -e CONTEXT=262144 \
    -e KV=int8 -e VISION=cpu -e GPU=0 -e STRATA_FATTN=1 -e STRATA_FDEC=1 $1 >/dev/null
  docker start $C >/dev/null
  for i in $(seq 200); do [ "$(docker inspect -f '{{.State.Health.Status}}' $C)" = healthy ] && break; sleep 3; done
}
trap 'docker rm -f $C >/dev/null 2>&1; docker start PolyStrata >/dev/null; echo "strata-p13 removed, PolyStrata started"' EXIT
for run in "$@"; do
  v=${run%%:*}; dir=${run#*:}
  case $v in base) img=polystrata:tower ;; b36) img=polystrata:tower-b36 ;; esac
  cat $F > /dev/null
  mk $img
  echo "== $v $dir $(date +%T) $img: $(docker exec $C sh -c "grep -E 'strata-poly: loaded' /opt/strata/strata-cybertiel-ud-q4_k_xl.log | tail -1")"
  echo "cached: $(fincore $F | tail -1 | awk '{print $1}')  $(awk '/MemAvailable/{print int($2/1048576) " GiB available"}' /proc/meminfo)"
  docker exec $C sh -c "cd /opt/strata && .venv/bin/python /data/profile/short12.py"
  docker exec $C sh -c "cd /opt/strata && .venv/bin/python /data/needle.py 4" | tail -1
  docker exec $C sh -c "cd /opt/strata && .venv/bin/python /data/multiturn.py $dir $TURNS"
  echo "after: cached $(fincore $F | tail -1 | awk '{print $1}'), engine RSS $(docker exec $C sh -c 'grep VmRSS /proc/$(pgrep -f engine-poly/strata-poly | head -1)/status'), $(docker exec $C sh -c "curl -s 127.0.0.1:8080/metrics" | grep -o '"expert_slots": [0-9]*\|"vram_free_mib": [0-9]*' | tr '\n' ' ')"
done 2>&1 | tee -a $D/pass13-ab.txt
