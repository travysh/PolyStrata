#!/bin/sh
# Pass 15: the Pass 14 trace slice replayed through cxb-p15 (cpu_experts.cpp of p15/trace15.patch, down by expert),
# for each argument "GAP_US:SPIN_US", ROUNDS interleaved rounds, PolyStrata stopped (started again at the end).
# GAP_US: the wait before each call (the GPU's part of a layer); SPIN_US: STRATA_CPU_SPIN_US (0: today's count).
D=/mnt/user/appdata/polystrata; IMG=strata-cybertiel:tower-5ca982fa; GG=/gguf/Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf
TRACE=${TRACE:-trace-slice.txt}; REP=${REP:-2}; ROUNDS=${ROUNDS:-3}
docker stop -t 60 PolyStrata >/dev/null 2>&1
trap 'docker start PolyStrata >/dev/null; echo "PolyStrata started"' EXIT
{
echo "== $(date) $TRACE x $REP, rounds $ROUNDS: $*"
for round in $(seq $ROUNDS); do
  for v in "$@"; do
    printf "round %d gap %s spin %s: " $round ${v%:*} ${v#*:}
    docker run --rm -e STRATA_CPU_DOWN_ORDER=1 -e CXB_GAP_US=${v%:*} -e STRATA_CPU_SPIN_US=${v#*:} -v $D/p15:/w \
      -v /mnt/cache/llm:/gguf --entrypoint sh $IMG -c "cd /w && ./cxb-p15 tab.txt $GG trace $TRACE $REP 0 12" 2>&1 \
      | grep -v '^strata-poly:' | tr '\n' ' '
    echo
  done
done
} 2>&1 | tee -a $D/data/pass15-${OUT:-bench}.txt
