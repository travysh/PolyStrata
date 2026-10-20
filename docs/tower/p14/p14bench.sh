#!/bin/sh
# Pass 14: the engine's recorded CPU calls (a slice of a STRATA_CPU_TRACE file) replayed through each cxb-<variant>,
# ROUNDS interleaved rounds, PolyStrata stopped (started again at the end).  Output: /data/pass14-${OUT:-bench}.txt
D=/mnt/user/appdata/polystrata; IMG=strata-cybertiel:tower-64f17cc2; GG=/gguf/Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf
TRACE=${TRACE:-trace-slice.txt}; REP=${REP:-2}; ROUNDS=${ROUNDS:-3}
docker stop -t 60 PolyStrata >/dev/null 2>&1
trap 'docker start PolyStrata >/dev/null; echo "PolyStrata started"' EXIT
{
echo "== $(date) $TRACE x $REP, rounds $ROUNDS: $*"
for round in $(seq $ROUNDS); do
  for v in "$@"; do
    printf "round %d %-8s " $round $v
    bin=${v%@*}; th=12; case $v in *@*) th=${v#*@} ;; esac   # variant@threads
    docker run --rm -v $D/p14:/w -v /mnt/cache/llm:/gguf --entrypoint sh $IMG -c "cd /w && ./cxb-$bin tab.txt $GG trace $TRACE $REP 0 $th"
  done
done
} 2>&1 | tee -a $D/data/pass14-${OUT:-bench}.txt
