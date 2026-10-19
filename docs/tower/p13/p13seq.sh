#!/bin/sh
# Pass 13: the Pass 12 sequence on the template container PolyStrata as it is: restart, 3 short requests, the 107K
# recall warm-up, multiturn.py $DIR $TURNS.  Output: /data/pass13-$1.txt (appended).
D=/mnt/user/appdata/polystrata/data; F=/mnt/cache/llm/Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf; C=PolyStrata
DIR=${DIR:-fwd}; TURNS=${TURNS:-3}
{
docker restart -t 60 $C >/dev/null
for i in $(seq 200); do [ "$(docker inspect -f '{{.State.Health.Status}}' $C)" = healthy ] && break; sleep 3; done
echo "== $1 $DIR $TURNS $(date +%T) $(docker inspect -f '{{.Config.Image}} {{.Image}}' $C | cut -c1-40): $(docker exec $C sh -c "grep -E 'STRATA_F|strata-poly: loaded|skip|stage' /opt/strata/strata-cybertiel-ud-q4_k_xl.log | tr '\n' ' '")"
echo "cached: $(fincore $F | tail -1 | awk '{print $1}')  $(awk '/MemAvailable/{print int($2/1048576) " GiB available"}' /proc/meminfo)"
docker exec $C sh -c "cd /opt/strata && .venv/bin/python /data/profile/short12.py"
docker exec $C sh -c "cd /opt/strata && .venv/bin/python /data/needle.py 4" | tail -1
docker exec $C sh -c "cd /opt/strata && .venv/bin/python /data/multiturn.py $DIR $TURNS"
echo "cached after: $(fincore $F | tail -1 | awk '{print $1}')  $(docker exec $C sh -c "curl -s 127.0.0.1:8080/metrics" | grep -o '"expert_slots": [0-9]*\|"vram_free_mib": [0-9]*' | tr '\n' ' ')"
} 2>&1 | tee -a $D/pass13-$1.txt
