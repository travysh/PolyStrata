#!/bin/sh
# Pass 15 engine runs (p14ab.sh with the PASS15 worker line and the CPU busy share during multiturn): each argument "label|image|extra docker args|dir|turns|engine args|needle TOKENS seed", e.g.
#   "base|polystrata:tower||fwd|5|"  "pf|polystrata:p14|-e STRATA_CPU_PREFETCH=1|rev|5|"  "w1|...||fwd|3|--window 1"
# A container strata-p13 with the PolyStrata template's settings (PolyStrata stopped, started again at the end), then
# 3 short requests, the 107K recall warm-up, multiturn.py dir turns, and the engine's per-request CPU-expert lines
# (measurement builds).  Engine args go into the config for that run and are taken out after.
D=/mnt/user/appdata/polystrata/data; F=/mnt/cache/llm/Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf; C=strata-p13
CFG=$D/config/strata-cybertiel-ud-q4_k_xl.json; REF=$CFG.q8-256k; LOG=/opt/strata/strata-cybertiel-ud-q4_k_xl.log
cmp -s $CFG $REF || { echo "config differs from .q8-256k, stopping"; exit 1; }
docker stop -t 60 PolyStrata >/dev/null 2>&1
trap 'cp -p $REF $CFG; docker rm -f $C >/dev/null 2>&1; docker start PolyStrata >/dev/null; echo "strata-p13 removed, config .q8-256k, PolyStrata started"' EXIT
for run in "$@"; do
  label=$(echo "$run" | cut -d'|' -f1); img=$(echo "$run" | cut -d'|' -f2); extra=$(echo "$run" | cut -d'|' -f3)
  dir=$(echo "$run" | cut -d'|' -f4); turns=$(echo "$run" | cut -d'|' -f5); eargs=$(echo "$run" | cut -d'|' -f6)
  ntok=$(echo "$run" | cut -d'|' -f7 | awk '{print $1}'); nseed=$(echo "$run" | cut -d'|' -f7 | awk '{print $2}')
  cp -p $REF $CFG
  if [ -n "$eargs" ]; then
    q=$(for a in $eargs; do printf '"%s", ' "$a"; done)
    sed -i "s|\"args\": \[|\"args\": [$q|" $CFG
  fi
  cat $F > /dev/null
  docker rm -f $C >/dev/null 2>&1
  docker create --name $C --gpus all -p 8080:8080 --ulimit memlock=-1 --stop-timeout 60 --restart no \
    -v $D:/data -v /mnt/cache/llm:/gguf -e FAMILY=cybertiel -e MODEL=UD-Q4_K_XL -e GGUF_DIR=/gguf -e CONTEXT=262144 \
    -e KV=int8 -e VISION=cpu -e GPU=0 -e STRATA_FATTN=1 -e STRATA_FDEC=1 $extra $img >/dev/null
  docker start $C >/dev/null
  for i in $(seq 200); do [ "$(docker inspect -f '{{.State.Health.Status}}' $C)" = healthy ] && break; sleep 3; done
  echo "== $label $dir $turns $(date +%T) $img [$extra] [$eargs]: $(docker exec $C sh -c "grep -E 'strata-poly: loaded|STRATA_CPU_' $LOG | tail -3 | tr '\n' ' '")"
  echo "cached: $(fincore $F | tail -1 | awk '{print $1}')  $(awk '/MemAvailable/{print int($2/1048576) " GiB available"}' /proc/meminfo)"
  docker exec $C sh -c "cd /opt/strata && .venv/bin/python /data/profile/short12.py"
  T0=$(date +%s)
  docker exec -e TOKENS=${ntok:-80000} $C sh -c "cd /opt/strata && .venv/bin/python /data/needle.py ${nseed:-4}" | tail -1
  echo "needle ${ntok:-80000} seed ${nseed:-4}: $(( $(date +%s) - T0 )) s"
  s0=$(head -1 /proc/stat)
  docker exec $C sh -c "cd /opt/strata && .venv/bin/python /data/multiturn.py $dir $turns"
  s1=$(head -1 /proc/stat)
  echo "$s0" "$s1" | awk '{for(i=2;i<=9;i++){a+=$i;b+=$(i+11)}; ia=$5+$6; ib=$16+$17; printf "CPU busy during multiturn: %.1f%% of 12 threads\n", 100*(1-(ib-ia)/(b-a))}'
  docker exec $C sh -c "grep -E 'PASS1[45]' $LOG" | sed 's/^/  /'
  echo "after: cached $(fincore $F | tail -1 | awk '{print $1}'), $(docker exec $C sh -c "curl -s 127.0.0.1:8080/metrics" | grep -o '"expert_slots": [0-9]*\|"vram_free_mib": [0-9]*' | tr '\n' ' ')"
done 2>&1 | tee -a $D/pass15-${OUT:-ab}.txt
