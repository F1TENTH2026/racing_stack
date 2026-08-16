#!/usr/bin/env bash
# slip_log_summary.sh — pulls kiss_icp_localization's slip-gate warnings
# (onOdom()'s erpm-vs-accelerometer cross-check, see localization_node.cpp)
# out of the most recent ROS2 log session and saves them for analysis.
set -euo pipefail
LOG_ROOT="${ROS_LOG_DIR:-$HOME/.ros/log}"
OUT="${1:-slip_log_$(date +%Y%m%d_%H%M%S).txt}"

SESSION_DIR=$(ls -td "$LOG_ROOT"/*/ 2>/dev/null | head -1) || true
if [ -z "$SESSION_DIR" ]; then
  echo "No ROS2 log sessions found under $LOG_ROOT" >&2
  exit 1
fi

# log filename follows the node's ROS name ("kiss_icp_localization", set in
# localization_node.cpp's Node() ctor), not the exec name (localization_node)
LOG_FILE=$(ls -t "$SESSION_DIR"kiss_icp_localization_*.log 2>/dev/null | head -1) || true
if [ -z "$LOG_FILE" ]; then
  LOG_FILE=$(ls -t "$SESSION_DIR"localization_node_*.log 2>/dev/null | head -1) || true
fi
if [ -z "$LOG_FILE" ]; then
  echo "No kiss_icp_localization log found in $SESSION_DIR" >&2
  exit 1
fi

echo "Session:  $SESSION_DIR"
echo "Log file: $LOG_FILE"

grep "slip gate" "$LOG_FILE" > "$OUT" || true
COUNT=$(wc -l < "$OUT")

echo "Slip-gate triggers: $COUNT"
echo "Saved to: $OUT"

if [ "$COUNT" -gt 0 ]; then
  echo
  echo "First trigger:"
  head -1 "$OUT"
  echo "Last trigger:"
  tail -1 "$OUT"
fi
