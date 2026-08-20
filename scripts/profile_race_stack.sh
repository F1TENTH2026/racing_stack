#!/usr/bin/env bash
set -u

DURATION_SEC="${1:-10}"
INTERVAL_SEC="${PROFILE_INTERVAL_SEC:-1}"

echo "race-stack profile: duration=${DURATION_SEC}s interval=${INTERVAL_SEC}s"
echo "timestamp pid cpu% rss_kib command"

end=$((SECONDS + DURATION_SEC))
while (( SECONDS < end )); do
  stamp=$(date -Iseconds)
  ps -eo pid=,pcpu=,rss=,comm= --sort=-pcpu | head -n 20 | \
    awk -v stamp="$stamp" '{print stamp, $0}'
  if command -v tegrastats >/dev/null 2>&1; then
    timeout 2 tegrastats --interval 1000 2>/dev/null | head -n 1 || true
  fi
  sleep "$INTERVAL_SEC"
done

echo "topic rates (5 second bounded samples)"
for topic in /scan /pf/pose/odom /odometry/local /car_state/odom /behavior_strategy /vesc/high_level/ackermann_cmd; do
  echo "$topic"
  timeout 5 ros2 topic hz "$topic" --window 100 2>&1 | tail -n 3 || true
done

echo "PF/controller deadline summaries are emitted by their nodes; preserve the ROS log with this report."
