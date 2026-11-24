#!/bin/sh
# pdl.sh URL OUT SIZE N: download in N ranges in parallel (resumable per part), join, rm parts
U=$1; O=$2; S=$3; N=$4; P=$(( (S + N - 1) / N ))
part() { i=$1; a=$((i*P)); b=$((a+P-1)); [ $b -ge $S ] && b=$((S-1)); want=$((b-a+1))
  for try in 1 2 3 4 5 6 7 8 9 10; do
    have=$(stat -c %s $O.p$i 2>/dev/null || echo 0); [ $have -ge $want ] && return 0
    curl -sfL -r $((a+have))-$b $U >> $O.p$i; done; return 1; }
i=0; while [ $i -lt $N ]; do part $i & i=$((i+1)); done; wait
i=0; : > $O; while [ $i -lt $N ]; do cat $O.p$i >> $O || exit 1; i=$((i+1)); done
[ $(stat -c %s $O) -eq $S ] && rm -f $O.p* && echo ok $O
