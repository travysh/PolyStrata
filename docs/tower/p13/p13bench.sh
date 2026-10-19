#!/bin/sh
# Pass 13 Phase 1: 3 rounds of every variant, interleaved, ROWS 1 and 2, POOL experts a layer.  PolyStrata stopped.
# Memory is compacted before each anonymous run so MADV_HUGEPAGE can get 2 MiB pages.  Output: /data/pass13-bench.txt
D=/mnt/user/appdata/polystrata/p13; IMG=${IMG:-strata-cybertiel:tower-64f17cc2}; POOL=${POOL:-128}; TOK=${TOK:-200}
GG=/gguf/Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf
{
echo "== $(date) pool $POOL tokens $TOK, THP $(cat /sys/kernel/mm/transparent_hugepage/enabled) defrag $(cat /sys/kernel/mm/transparent_hugepage/defrag)"
for round in 1 2 3; do
  for rows in 1 2; do
    for v in t:mmap t:anon4k t:anon2m b:mmap b:anon2m t:repack; do
      bin=${v%%:*}; mode=${v#*:}
      [ $mode != mmap ] && { sync; echo 1 > /proc/sys/vm/compact_memory; }
      printf "round %d cxb-%s " $round $bin
      docker run --rm -v $D:/w -v /mnt/cache/llm:/gguf --entrypoint sh $IMG -c "cd /w && ./cxb-$bin tab.txt $GG $mode $rows $TOK $POOL"
    done
  done
done
} 2>&1 | tee -a /mnt/user/appdata/polystrata/data/pass13-bench.txt
