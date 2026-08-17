#!/usr/bin/env bash
# cpu_profile.sh — lightweight per-process CPU/mem sampler for diagnosing
# racing_stack CPU contention on the real car.
#
# Run this in its own terminal alongside race.launch.xml (and joy_node)
# BEFORE you start driving, and stop it with Ctrl+C once you're done. It
# just wraps `top -b`, so the sampler itself adds negligible overhead —
# safe to run during an actual drive session.
#
# Usage:
#   ./cpu_profile.sh [output_file] [interval_seconds]
#
# Then hand the resulting file back for analysis. If you can note roughly
# when in the session something went wrong (elapsed time, or which lap),
# that helps line up CPU spikes with the actual symptom.

set -euo pipefail

OUT="${1:-cpu_profile_$(date +%Y%m%d_%H%M%S).log}"
INTERVAL="${2:-2}"

echo "Logging per-process CPU/mem to: $OUT"
echo "Sampling every ${INTERVAL}s. Ctrl+C to stop when the drive session is done."

# -c: full command line (needed to tell same-named python3 nodes apart)
# -w 512: wide output so long ros2 node command lines aren't truncated
# -b -d: batch mode, sample every INTERVAL seconds, until killed
top -b -c -w 512 -d "$INTERVAL" > "$OUT"
