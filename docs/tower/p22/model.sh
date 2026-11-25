#!/bin/sh
# p22/model.sh MODEL LABEL SAMPLES [ids...]: start MODEL (cybertiel or inkling) in the test container (port 5003, live's
# environment, image $IMAGE) and run run_bench.py on the tasks: answers in p22/runs/LABEL.jsonl, log in p22/runs/LABEL.log.
# TASKS: the task folder under p22 (default tasks; think.sh uses think-tasks).
# Stops live PolyStrata first (one card); does not start it again (all.sh does).
M=$1; L=$2; N=$3; shift 3; D=/mnt/user/appdata/polystrata; I=${IMAGE:-polystrata:next-f6bc1acf}
case $M in
  cybertiel) C=/data/config/strata-cybertiel-next.json ;;
  inkling)   C=/data/config/strata-inkling2-mini.json ;;
  *) echo "model: cybertiel or inkling"; exit 2 ;;
esac
C=${CONFIG:-$C}                                   # CONFIG: another config for the same model (a test copy)
mkdir -p $D/p22/runs
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
docker run -d --name polystrata-next --gpus all -p 5003:8080 --ulimit memlock=-1 --stop-timeout 60 --restart no \
  -v /mnt/cache/llm:/gguf -v $D/data:/data -v $D/p22:/p22 -v /mnt/cache/appdata/polystrata/park:/park \
  -e TMPDIR=/park -e STRATA_POLY_MIRROR=0 -e STRATA_CPU_NO_X4=1 -e TZ=America/Los_Angeles $I \
  .venv/bin/python serve/server.py --engine strata --config $C --port 8080 --gpu 0 >/dev/null
until curl -s -m 5 localhost:5003/health | grep -q "\"loaded\": true"; do
  sleep 5; docker ps -q -f name=polystrata-next | grep -q . || { echo "container exited" >> $D/p22/runs/$L.log; exit 1; }
done
echo "== $L: $M on $I, $C, $(date)" >> $D/p22/runs/$L.log
for p in $PROBE; do                               # PROBE: request files in p22/ sent once first; their prompt size is logged
  echo "probe $p: $(curl -s -m 600 localhost:5003/v1/chat/completions -H 'Content-Type: application/json' -d @$D/p22/$p | grep -o '"prompt_tokens": [0-9]*')" >> $D/p22/runs/$L.log
done
docker exec polystrata-next sh -c "cd /opt/strata/terminal && .venv/bin/python /p22/run_bench.py /p22/${TASKS:-tasks} /p22/runs/$L.jsonl $N $*" >> $D/p22/runs/$L.log 2>&1
echo "== done $(date)" >> $D/p22/runs/$L.log
docker rm -f polystrata-next >/dev/null
