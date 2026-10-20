# engine VmRSS, page cache and park-dir usage every 0.5 s, with the count of conversation cache log lines so far
L=/opt/strata/strata-cybertiel-ud-q4_k_xl.log
S=$(wc -l < $L)
while true; do
  P=$(pgrep -f engine-poly/strata-poly | head -1)
  echo "$(date +%s.%N | cut -c1-14) rss_kb=$(awk "/VmRSS/{print \$2}" /proc/$P/status) cached_kb=$(awk "/^Cached:/{print \$2}" /proc/meminfo) park_used_kb=$(df -k /park | awk "NR==2{print \$3}") events=$(tail -n +$((S+1)) $L | grep -c "conversation cache")"
  sleep 0.5
done
