#!/bin/sh
# p22/all.sh TAG SAMPLES [ids...]: both models (Cyber-Tiel first), then grading with no network, then live back.
T=$1; N=$2; shift 2; D=/mnt/user/appdata/polystrata; cd $D/p22
trap 'docker rm -f polystrata-next >/dev/null 2>&1; docker start PolyStrata >/dev/null; date > runs/$T.done' EXIT
./model.sh cybertiel $T-cybertiel $N "$@"
./model.sh inkling $T-inkling $N "$@"
for m in cybertiel inkling; do
  docker run --rm --network none --cpus 6 -v $D/p22:/p22 -w /p22 --entrypoint /opt/strata/terminal/.venv/bin/python \
    ${IMAGE:-polystrata:next-f6bc1acf} grade.py grade /p22/tasks /p22/runs/$T-$m.jsonl > runs/$T-$m.grade.log 2>&1
done
