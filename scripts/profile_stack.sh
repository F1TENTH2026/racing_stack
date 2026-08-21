#!/usr/bin/env bash
set -u

duration="${1:-10}"
out_dir="${2:-/tmp/racing_stack_profile_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$out_dir/topics"

topics=(
  /vesc/odom /vesc/sensors/imu /scan /pf/pose/odom /odometry/local
  /car_state/odom /detect/raw_obstacles /tracking/obstacles
  /behavior_strategy /vesc/high_level/ackermann_cmd
)

printf 'duration_sec=%s\nstarted=%s\n' "$duration" "$(date --iso-8601=seconds)" > "$out_dir/meta.txt"
ps -eo pid,psr,pcpu,rss,comm,args --sort=-pcpu > "$out_dir/processes_start.txt"

pids=()
for topic in "${topics[@]}"; do
  safe_name="${topic//\//_}"
  timeout --signal=INT "$duration" ros2 topic hz "$topic" --window 10000 \
    > "$out_dir/topics/${safe_name}.txt" 2>&1 &
  pids+=("$!")
done

if command -v tegrastats >/dev/null 2>&1; then
  timeout --signal=INT "$duration" tegrastats --interval 1000 > "$out_dir/tegrastats.txt" 2>&1 &
  pids+=("$!")
fi

for pid in "${pids[@]}"; do
  wait "$pid" || true
done
ps -eo pid,psr,pcpu,rss,comm,args --sort=-pcpu > "$out_dir/processes_end.txt"
printf 'finished=%s\noutput=%s\n' "$(date --iso-8601=seconds)" "$out_dir" | tee -a "$out_dir/meta.txt"
