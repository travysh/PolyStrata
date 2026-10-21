#!/bin/sh
# Pass 17: the engine's recorded CPU calls (Pass 14's 60,000-call slice) replayed through each cxb-<variant> of
# p17/cpu_experts_v.cpp, after one warm-up pass, ROUNDS interleaved rounds, PolyStrata stopped (started again at the
# end).  A variant named p14:<name> runs Pass 14's p14/cxb-<name>.  Output: /data/pass17-${OUT:-split}.txt
#   sh p17bench.sh full arith reads idle
D=/mnt/user/appdata/polystrata; IMG=strata-cybertiel:tower-64f17cc2; GG=/gguf/Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf
TRACE=${TRACE:-trace-slice.txt}; REP=${REP:-2}; ROUNDS=${ROUNDS:-3}
docker stop -t 60 PolyStrata >/dev/null 2>&1
trap 'docker start PolyStrata >/dev/null; echo "PolyStrata started"' EXIT
{
echo "== $(date) $TRACE x $REP, rounds $ROUNDS: $*"
for round in $(seq $ROUNDS); do
  for v in "$@"; do
    printf "round %d %-8s " $round $v
    dir=p17; bin=$v; case $v in p14:*) dir=p14; bin=${v#p14:} ;; esac
    docker run --rm -v $D/$dir:/w -v $D/p17:/p17 -v /mnt/cache/llm:/gguf --entrypoint sh $IMG -c \
      "cd /w && ./cxb-$bin /p17/tab.txt $GG trace /p17/$TRACE $REP 0 12"
  done
done
} 2>&1 | tee -a $D/data/pass17-${OUT:-split}.txt
