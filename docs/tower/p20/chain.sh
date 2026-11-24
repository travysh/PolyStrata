#!/bin/sh
docker stop PolyStrata >/dev/null
for r in new-a:polystrata:next-aadae31e old-a:polystrata:tower-c868922e new-b:polystrata:next-aadae31e old-b:polystrata:tower-c868922e; do
  ./t.sh ${r%%:*} ${r#*:}
done
docker rm -f polystrata-next >/dev/null
docker start PolyStrata >/dev/null
date > chain.done
