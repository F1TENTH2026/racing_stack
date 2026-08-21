#!/usr/bin/env bash
set -u

duration="${1:-10}"
out_dir="${2:-/tmp/racing_stack_profile_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$out_dir"

printf 'duration_sec=%s\nstarted=%s\n' "$duration" "$(date --iso-8601=seconds)" > "$out_dir/meta.txt"
ps -eo pid,psr,pcpu,rss,comm,args --sort=-pcpu > "$out_dir/processes_start.txt"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$script_dir/profile_topics.py" --duration "$duration" \
  --output "$out_dir/topic_rates.json" > "$out_dir/topic_sampler.txt" 2>&1 &
pids=("$!")

if command -v tegrastats >/dev/null 2>&1; then
  timeout --signal=INT "$duration" tegrastats --interval 1000 > "$out_dir/tegrastats.txt" 2>&1 &
  pids+=("$!")
fi

for pid in "${pids[@]}"; do
  wait "$pid" || true
done
ps -eo pid,psr,pcpu,rss,comm,args --sort=-pcpu > "$out_dir/processes_end.txt"
printf 'finished=%s\noutput=%s\n' "$(date --iso-8601=seconds)" "$out_dir" | tee -a "$out_dir/meta.txt"
