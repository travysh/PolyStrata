#!/bin/sh
# Pass 13 Phase 1, second part.  (1) today / PR 36 / PR 36's prefetch alone (down tasks of 16) at 1-4 tokens a
# call, from the mmap.  (2) huge pages at full coverage: the page cache dropped and memory compacted first.
D=/mnt/user/appdata/polystrata/p13; IMG=strata-cybertiel:tower-64f17cc2; GG=/gguf/Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf
r() { docker run --rm -v $D:/w -v /mnt/cache/llm:/gguf --entrypoint sh $IMG -c "cd /w && ./cxb-$1 tab.txt $GG $2 $3 200 128"; }
{
echo "== $(date) part 2"
for round in 1 2 3; do for rows in 1 2 3 4; do for b in t b p; do printf "round %d cxb-%s " $round $b; r $b mmap $rows; done; done; done
for round in 1 2 3; do for rows in 1 2; do
  for m in anon4k anon2m; do
    sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory
    printf "round %d dropped cxb-t " $round; r t $m $rows
  done
done; done
} 2>&1 | tee -a /mnt/user/appdata/polystrata/data/pass13-bench2.txt
