#!/bin/sh
# runbench.sh LABEL PY: the comparison set for the 0.2.19 update, inside a container; results to /data/upd-LABEL.txt
L=$1; PY=$2; O=/data/upd-$L.txt
{
echo "== $L $(date)"
$PY /data/decode.py $L-decode 4
$PY /data/bench_11k.py $L-11k
$PY /data/aba.py $L-aba 24000 48000
pid=$(pgrep -f "strata-poly --serve" | head -1)
grep -E "VmRSS|RssAnon|RssFile" /proc/$pid/status
echo "== done $(date)"
} > $O 2>&1
