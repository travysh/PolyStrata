#!/bin/sh
docker stop PolyStrata >/dev/null
./o.sh v145-a; ./t.sh new-c polystrata:next-aadae31e; ./o.sh v145-b; ./t.sh new-d polystrata:next-aadae31e
docker rm -f polystrata-next >/dev/null
docker start PolyStrata >/dev/null
date > chain2.done
