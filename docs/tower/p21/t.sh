#!/bin/sh
# p21/t.sh LABEL IMAGE [ENV=VAL ...]: Inkling 2 Mini in the test container (port 5003) with live's environment, after
# p20/t.sh.  CFG=path picks another config.  Stops live PolyStrata first (one card).  Results: p21/LABEL.txt.  Scripts in /data/p21 and /data.
L=$1; I=$2; shift 2; E=""; for kv in "$@"; do E="$E -e $kv"; done; D=/mnt/user/appdata/polystrata; O=$D/p21/$L.txt
C=${CFG:-/data/config/strata-inkling2-mini.json}; TOK=/data/packs/inkling-ud-q4_k_xl/tokenizer
docker stop PolyStrata >/dev/null 2>&1; docker rm -f polystrata-next >/dev/null 2>&1
docker run -d --name polystrata-next --gpus all -p 5003:8080 --ulimit memlock=-1 --stop-timeout 60 --restart no \
  -v /mnt/cache/llm:/gguf -v $D/data:/data -v /mnt/cache/appdata/polystrata/park:/park \
  -e TMPDIR=/park -e STRATA_POLY_MIRROR=0 -e STRATA_CPU_NO_X4=1 -e TZ=America/Los_Angeles $E $I \
  .venv/bin/python serve/server.py --engine strata --config $C --port 8080 --gpu 0 >/dev/null
docker cp /tmp/olddocs polystrata-next:/opt/strata/docs; docker cp /tmp/oldREADME.md polystrata-next:/opt/strata/README.md
t0=$(date +%s)
until curl -s -m 5 localhost:5003/health | grep -q "\"loaded\": true"; do
  sleep 5; docker ps -q -f name=polystrata-next | grep -q . || { echo "container exited" > $O; exit 1; }
done
P="cd /opt/strata/terminal && PYTHONPATH=/opt/strata/terminal"; X="docker exec polystrata-next sh -c"
{
echo "== $L $I $(date), loaded in $(( $(date +%s) - t0 )) s"
$X "grep -hE 'experts|VRAM|requant|loop|tail' /opt/strata/terminal/strata-inkling*.log | head -20"
$X "$P .venv/bin/python /data/decode.py warm 2 >/dev/null"
$X "$P .venv/bin/python /data/p21/features.py http://127.0.0.1:8080 $TOK"
$X "$P .venv/bin/python /data/p21/stream.py"
$X "$P .venv/bin/python -m unittest serve.test_grammar serve.test_logit_bias serve.test_prefill 2>&1 | tail -3"
$X "$P .venv/bin/python /data/decode.py $L-decode 4"
$X "$P .venv/bin/python /data/bench_11k.py $L-11k"
$X "$P .venv/bin/python /data/aba.py $L-aba 24000"
$X "$P .venv/bin/python /data/ldecode.py $L-ldec 8000 24000"
$X 'grep -E "VmRSS|RssAnon|RssFile" /proc/$(pgrep -f "strata-poly --serve" | head -1)/status'
$X "grep -hiE 'error|warn|from VRAM' /opt/strata/terminal/strata-inkling*.log | sort | uniq -c | sort -rn | head -20"
echo "== done $(date)"
} > $O 2>&1
