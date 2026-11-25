#!/bin/sh
# p22/think.sh TAG SAMPLES [ids...]: the thinking puzzles (think-tasks/) on both models, Cyber-Tiel first, then live
# PolyStrata started again.  Grading runs on any machine with python3: think.py grade think-tasks runs/TAG-*.jsonl
T=$1; N=$2; shift 2; cd /mnt/user/appdata/polystrata/p22
trap 'docker rm -f polystrata-next >/dev/null 2>&1; docker start PolyStrata >/dev/null; date > runs/$T.done' EXIT
TASKS=think-tasks ./model.sh cybertiel $T-cybertiel $N "$@"
TASKS=think-tasks ./model.sh inkling $T-inkling $N "$@"
