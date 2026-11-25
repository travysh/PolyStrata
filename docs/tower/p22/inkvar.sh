#!/bin/sh
# p22/inkvar.sh TAG: the pictures (img-tasks) on Cyber-Tiel, then Inkling with three configs (setup's; a picture limit
# of 1,024 tokens; that and the small-card settings off), then live PolyStrata again.  Each run first logs the prompt
# size of two image questions sent without their picture (probe-*.json), to count how many tokens a picture takes.
T=$1; cd /mnt/user/appdata/polystrata/p22
trap 'docker rm -f polystrata-next >/dev/null 2>&1; docker start PolyStrata >/dev/null; date > runs/$T.done' EXIT
export TASKS=img-tasks PROBE="probe-chart5.json probe-space3.json"
./model.sh cybertiel $T-cybertiel 1
./model.sh inkling $T-ink300 1
CONFIG=/data/config/p22-ink-v1024.json ./model.sh inkling $T-ink1024 1
CONFIG=/data/config/p22-ink-acc1024.json ./model.sh inkling $T-inkacc1024 1
