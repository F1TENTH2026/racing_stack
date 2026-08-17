#!/bin/bash
# 3개 노드 함수별 CPU 프로파일 -> cpu_log.txt
# usage: ./prof3.sh [duration_sec]

DUR=${1:-30}
OUT=cpu_log.txt
NODES=(recovery_spliner_node controller_manager state_machine)

echo "===== $(date '+%F %T')  duration=${DUR}s =====" >> "$OUT"

for N in "${NODES[@]}"; do
  PID=$(pgrep -f "$N" | head -1)
  if [ -z "$PID" ]; then
    echo "[$N] not running" >> "$OUT"
    continue
  fi
  (
    RAW=$(mktemp)
    py-spy record --pid "$PID" --duration "$DUR" --format raw -F -o "$RAW" 2>/dev/null
    TOTAL=$(awk '{s+=$NF} END{print s}' "$RAW")
    {
      echo ""
      echo "----- [$N] pid=$PID  samples=$TOTAL -----"
      # 스택 맨 끝 프레임(실제 실행 중인 함수)으로 집계
      awk -v tot="$TOTAL" '{
        n=$NF; $NF=""
        m=split($0, a, ";"); leaf=a[m]
        gsub(/^ +| +$/, "", leaf)
        c[leaf]+=n
      } END {
        for (f in c) printf "%6.2f%%  %8d  %s\n", 100*c[f]/tot, c[f], f
      }' "$RAW" | sort -rn | head -25
    } >> "$OUT"
    rm -f "$RAW"
  ) &
done

wait
echo "done -> $OUT"
