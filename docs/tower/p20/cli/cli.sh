#!/bin/sh
# cli.sh LABEL IMAGE PROMPT [FOLLOW] [ENV=VAL ...]: the engine alone (one-shot mode) on PROMPT.ids, 300 tokens.
# FOLLOW "-" for none, else a token file to write in place of the picks (no drafts then). Fresh profile copy per run.
T="${THREADS:+--threads $THREADS}"; L=$1; I=$2; P=$3; F=$4; shift 4; E=""; for kv in "$@"; do E="$E -e $kv"; done
D=/mnt/user/appdata/polystrata/data; W=/data/p20cli
case "$I" in *893468a5) X=/opt/strata/engine-poly/strata-poly ;; *) X=/opt/strata/terminal/engine-poly/strata-poly ;; esac
cp $D/packs/cybertiel-ud-q4_k_xl/expert-profile.txt $D/p20cli/profile-$L.txt
FA=""; [ "$F" != "-" ] && FA="--follow $(cat $D/p20cli/$F)"
docker run --rm --gpus all --ulimit memlock=-1 -v /mnt/cache/llm:/gguf -v $D:/data -e TZ=America/Los_Angeles $E \
  --entrypoint $X $I --vram-reserve-mib 512 --model /gguf/Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf \
  --max-context 262144 --expert-profile $W/profile-$L.txt --kv q8_0 --max-new 300 $T \
  --tokens "$(cat $D/p20cli/$P.ids)" $FA > $D/p20cli/$L.out 2> $D/p20cli/$L.err
echo "$L: $(grep -E "generated .* tok/s" $D/p20cli/$L.err | tail -1)"
