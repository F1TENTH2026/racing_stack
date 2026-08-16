#!/usr/bin/env bash
#
# gp_chain_check.sh -- GP opponent-prediction chain health + CPU capture.
#
# Purpose: verify the opponent_trajectory busy-loop fix. Run it once BEFORE the
# fix and once AFTER, with the same scenario, then diff the two logs.
#
#   opponent_trajectory  -> /proj_opponent_trajectory
#     -> gaussian_process_opp_traj -> /opponent_trajectory
#       -> opp_prediction          -> /opponent_prediction/obstacles_pred
#
# Two things must hold for the fix to be accepted:
#   1. the three chain topics publish at the SAME rate as before  (no regression)
#   2. opponent_trajectory's CPU collapses when no opponent is in sight (the win)
#
# Usage:
#   ./gp_chain_check.sh [tag] [duration_sec]
#     tag           label for this run, e.g. before | after   (default: run)
#     duration_sec  measurement window                        (default: 20)
#
# Writes  <repo>/logfile/gp_chain_<tag>_<stamp>.log  and refreshes the
# latest_gp_chain_<tag>.log symlink, matching the naming the state_machine and
# static_avoidance debug logs already use in that directory.
#
# Not installed by CMakeLists (which lists scripts explicitly); run it from the
# source tree. Needs a sourced ROS 2 environment. No sysstat/pidstat dependency:
# CPU is read straight from /proc, which also bounds it to exactly the window.

set -uo pipefail

TAG="${1:-run}"
DURATION="${2:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$REPO_ROOT/logfile"

TOPICS=(
  /proj_opponent_trajectory
  /opponent_trajectory
  /opponent_prediction/obstacles_pred
)

# opponent_trajectory is the node under test. The path prefix keeps pgrep from
# also matching predictor_opponent_trajectory.
NODE_PATTERN='gp_traj_predictor/opponent_trajectory'

# ---------------------------------------------------------------------------

if ! command -v ros2 >/dev/null 2>&1; then
    echo "error: 'ros2' not found -- source your ROS 2 environment first." >&2
    exit 1
fi

mkdir -p "$LOG_DIR" || { echo "error: cannot create $LOG_DIR" >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/gp_chain_${TAG}_${STAMP}.log"
LATEST="$LOG_DIR/latest_gp_chain_${TAG}.log"

CLK_TCK="$(getconf CLK_TCK 2>/dev/null || echo 100)"

# utime+stime for a pid, in clock ticks. Strips "pid (comm) " first so a comm
# containing spaces cannot shift the field offsets.
cpu_ticks() {
    local raw
    raw="$(cat "/proc/$1/stat" 2>/dev/null)" || return 1
    raw="${raw#*) }"        # remaining field 12 = utime, 13 = stime
    awk '{print $12+$13}' <<<"$raw"
}

log() { printf '%s\n' "$*" >>"$LOG"; }

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
{
    echo "==============================================================="
    echo " GP chain check -- tag=$TAG  window=${DURATION}s"
    echo "==============================================================="
    echo "date      : $(date -Is)"
    echo "host      : $(hostname)"
    echo "kernel    : $(uname -r)  arch=$(uname -m)  nproc=$(nproc)"
    echo "ros distro: ${ROS_DISTRO:-<unset>}"
    echo "rmw       : ${RMW_IMPLEMENTATION:-<default>}"
    echo "domain id : ${ROS_DOMAIN_ID:-0}"
} >"$LOG"

if git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    dirty=""
    [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ] && dirty=" (DIRTY)"
    log "git       : $(git -C "$REPO_ROOT" rev-parse --short HEAD)$dirty on $(git -C "$REPO_ROOT" branch --show-current)"
    log "git subj  : $(git -C "$REPO_ROOT" log -1 --pretty=%s)"
fi

log ""
log "NOTE: each 'ros2 topic hz' is itself a DDS participant; 3 extra"
log "      participants are live during this window. Keep that constant"
log "      between the before/after runs so the comparison stays fair."

# ---------------------------------------------------------------------------
# Node under test
# ---------------------------------------------------------------------------
log ""
log "--- node under test ---------------------------------------------"
PID="$(pgrep -f "$NODE_PATTERN" | head -1)"
if [ -z "$PID" ]; then
    log "opponent_trajectory : NOT RUNNING"
    log "  -> topic rates below will be empty; start the stack first."
else
    log "opponent_trajectory : pid $PID"
    log "  cmdline: $(tr '\0' ' ' <"/proc/$PID/cmdline" 2>/dev/null)"
    log "  threads: $(awk '/^Threads:/{print $2}' "/proc/$PID/status" 2>/dev/null)"
fi

log ""
log "--- system before ------------------------------------------------"
log "uptime : $(uptime)"
log "memory : $(free -h | awk '/^Mem:/{printf "used %s / %s", $3, $2}')"

# ---------------------------------------------------------------------------
# Measurement window: all three hz probes in parallel so they cover the SAME
# window (they are one chain -- staggering them would be misleading), with the
# CPU sample bracketing exactly that window.
# ---------------------------------------------------------------------------
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "measuring for ${DURATION}s ..." >&2

for t in "${TOPICS[@]}"; do
    # PYTHONUNBUFFERED so the lines are flushed before timeout kills the probe;
    # stdbuf does not reliably defeat Python's own buffering layer.
    # shellcheck disable=SC2024
    timeout -s INT "$DURATION" env PYTHONUNBUFFERED=1 \
        ros2 topic hz "$t" >"$TMP/$(echo "$t" | tr '/' '_').out" 2>&1 &
done

t0=$(date +%s.%N)
ticks0=""
[ -n "$PID" ] && ticks0="$(cpu_ticks "$PID")"

wait

t1=$(date +%s.%N)
ticks1=""
[ -n "$PID" ] && ticks1="$(cpu_ticks "$PID")"

# ---------------------------------------------------------------------------
# Topic rates
# ---------------------------------------------------------------------------
log ""
log "--- topic rates (window ${DURATION}s) ----------------------------"
SUMMARY=()
for t in "${TOPICS[@]}"; do
    f="$TMP/$(echo "$t" | tr '/' '_').out"
    log ""
    log "[$t]"
    if [ -s "$f" ]; then
        sed 's/^/  /' "$f" >>"$LOG"
    else
        log "  (no output)"
    fi
    # Classify on the presence of a rate, not on file size: 'ros2 topic hz'
    # prints a "does not appear to be published yet" warning for a silent topic,
    # which is non-empty output but still means no data.
    avg="$(grep -o 'average rate: [0-9.]*' "$f" 2>/dev/null | tail -1 | awk '{print $3}')"
    if [ -n "$avg" ]; then
        SUMMARY+=("$t : $avg Hz")
    else
        SUMMARY+=("$t : NO DATA -- nothing published during the window")
    fi
done

# ---------------------------------------------------------------------------
# CPU of the node under test, over exactly the window above
# ---------------------------------------------------------------------------
log ""
log "--- opponent_trajectory CPU --------------------------------------"
CPU_LINE="n/a (node not running)"
if [ -n "$PID" ] && [ -n "$ticks0" ] && [ -n "$ticks1" ]; then
    CPU_LINE="$(awk -v a="$ticks0" -v b="$ticks1" -v t0="$t0" -v t1="$t1" -v hz="$CLK_TCK" \
        'BEGIN{ el=t1-t0; if(el<=0){print "n/a"; exit}
                printf "%.1f%% of one core  (cpu %.2fs over %.2fs wall)", (b-a)/hz/el*100, (b-a)/hz, el }')"
elif [ -n "$PID" ]; then
    CPU_LINE="n/a (process exited during the window)"
fi
log "$CPU_LINE"
log ""
log "PASS if this is under ~5% with no opponent in sight."
log "Before the fix this loop had no sleep on the opponent-absent path and"
log "sat at roughly a full core."

# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
log ""
log "--- system after -------------------------------------------------"
log "uptime : $(uptime)"

log ""
log "--- top 15 racing-stack processes by CPU -------------------------"
{
    ps -eo pcpu,pmem,pid,args --sort=-pcpu \
        | awk 'NR==1 || (/roboracer_ws/ && !/gp_chain_check|[a]wk |[p]s -eo/)' \
        | head -16 \
        | cut -c1-160
} >>"$LOG" 2>&1

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log ""
log "=================== SUMMARY (tag=$TAG) ==========================="
for s in "${SUMMARY[@]}"; do log "  $s"; done
log "  opponent_trajectory CPU : $CPU_LINE"
log "=================================================================="

ln -sfn "$(basename "$LOG")" "$LATEST"

# Echo the summary to the terminal too.
sed -n '/=================== SUMMARY/,$p' "$LOG"
echo
echo "log: $LOG"
echo "     $LATEST -> $(basename "$LOG")"
