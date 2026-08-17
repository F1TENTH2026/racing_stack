#!/bin/bash
# cpu_profile.sh — py-spy 로 파이썬 노드들의 함수별 CPU 점유를 뜬다.
#
# race.launch.xml 이 debug:=on 일 때 자동으로 띄운다. 터미널을 하나 더 열어
# 손으로 돌릴 필요가 없고, 결과는 state_machine / recovery_spliner 디버그 로그와
# 같은 자리(<repo>/logfile/)에 같은 규칙으로 떨어진다:
#
#     logfile/cpu_profile_<YYYYmmdd_HHMMSS>.log
#     logfile/latest_cpu_profile.log -> 위 파일
#
# 직접 돌릴 수도 있다:  ./cpu_profile.sh [duration] [delay] [node ...]
#
# 주의 1) py-spy 가 남의 프로세스에 붙으려면 ptrace 권한이 필요하다. 대부분의
#         우분투는 kernel.yama.ptrace_scope=1 이라 막힌다:
#             sudo sysctl kernel.yama.ptrace_scope=0          (재부팅하면 원복)
#             echo 'kernel.yama.ptrace_scope = 0' | sudo tee /etc/sysctl.d/10-ptrace.conf
#         권한이 없으면 이 스크립트는 그 사실을 로그에 남기고 조용히 끝난다.
#         런치를 실패시키지 않는다 — 프로파일러 때문에 주행이 막히면 안 된다.
#
# 주의 2) 프로파일링 자체가 CPU 를 쓴다. 6코어 Orin 에서 측정 대상이 많을수록
#         측정값이 오염되므로 노드 수를 늘릴 때는 감안할 것. debug:=off 에서는
#         아예 실행되지 않는다.
#
# 주의 3) 기본으로 --nonblocking 을 쓴다(대상 프로세스를 멈추지 않는다). 주행 중
#         제어 노드가 멈추면 VESC 페일세이프가 걸릴 수 있기 때문이다. 표본이 조금
#         어긋나는 것을 감수하는 대신 안전을 택한 것. 차를 세워두고 정확한 값이
#         필요하면 RACE_PROFILE_BLOCKING=1 로 끈다.

set -uo pipefail

DUR="${1:-30}"        # 표본 수집 시간 [s]
DELAY="${2:-20}"      # 노드가 다 뜨고 주행이 시작될 때까지 기다리는 시간 [s]
shift 2 2>/dev/null || true
NODES=("$@")
if [ ${#NODES[@]} -eq 0 ]; then
    # 지금까지 htop 에서 상위를 차지하던 파이썬 노드들.
    NODES=(state_machine controller_manager recovery_spliner_node
           static_avoidance_node change_avoidance_node tracking_node)
fi

# <repo>/logfile — 이 스크립트는 설치되면 share/stack_master/scripts/ 에 놓이므로
# 소스 트리의 위치로 되짚어 올라갈 수 없다. 노드들이 쓰는 것과 같은 규칙으로,
# RACE_DEBUG_LOG_DIR 을 우선 보고 없으면 git 체크아웃을 찾아 올라간다.
resolve_log_dir() {
    if [ -n "${RACE_DEBUG_LOG_DIR:-}" ]; then
        printf '%s' "$RACE_DEBUG_LOG_DIR"; return
    fi
    local d
    d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    while [ "$d" != "/" ]; do
        [ -d "$d/.git" ] && { printf '%s' "$d/logfile"; return; }
        d="$(dirname "$d")"
    done
    # 설치본에서 실행된 경우: 워크스페이스의 소스 트리를 흔한 위치에서 찾아본다.
    for c in "$HOME/roboracer_ws/src/racing_stack/logfile"; do
        [ -d "$c" ] && { printf '%s' "$c"; return; }
    done
    printf '%s' "$HOME/racing_logs"
}

LOG_DIR="$(resolve_log_dir)"
mkdir -p "$LOG_DIR" || exit 0          # 로그를 못 써도 주행은 계속되어야 한다
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$LOG_DIR/cpu_profile_${STAMP}.log"
ln -sfn "$(basename "$OUT")" "$LOG_DIR/latest_cpu_profile.log" 2>/dev/null

{
    echo "# cpu_profile 시작 $STAMP  (수집 ${DUR}s, 지연 ${DELAY}s)"
    echo "# host=$(hostname)  nproc=$(nproc)"
    echo "# 대상: ${NODES[*]}"
} > "$OUT"

if ! command -v py-spy >/dev/null 2>&1; then
    echo "# py-spy 없음 — 프로파일 건너뜀 (thirdimpact 환경에서 실행했는지 확인)" >> "$OUT"
    exit 0
fi

# ptrace 가 막혀 있으면 py-spy 는 붙자마자 실패한다. 30초를 기다렸다가 빈 로그를
# 남기느니 지금 알려주는 편이 낫다.
SCOPE="$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo 0)"
if [ "$SCOPE" != "0" ] && [ "$(id -u)" != "0" ]; then
    {
        echo "# ptrace_scope=$SCOPE 이고 root 가 아니라 py-spy 가 다른 프로세스에"
        echo "# 붙을 수 없다. 프로파일을 건너뛴다. 한 번만 풀어주면 된다:"
        echo "#     sudo sysctl kernel.yama.ptrace_scope=0"
    } >> "$OUT"
    exit 0
fi

# --nonblocking: 표본을 뜰 때 대상 프로세스를 멈추지 않는다. 기본 동작은 매
# 표본마다 프로세스를 잠깐 정지시키는데, VESC 는 상위 제어기의 명령이 일정 시간
# 끊기면 페일세이프로 모터 출력을 끊으므로 주행 중에는 정지가 없는 편이 안전하다.
# 대가로 스택을 읽는 도중 프로세스가 움직여 표본이 조금 어긋날 수 있다.
# 차를 세워두고 정확도를 우선하고 싶으면 RACE_PROFILE_BLOCKING=1 로 끈다.
NONBLOCK=(--nonblocking)
[ "${RACE_PROFILE_BLOCKING:-0}" = "1" ] && NONBLOCK=()

# 노드가 뜨고, 차가 실제로 달리기 시작한 뒤에 표본을 떠야 의미가 있다.
sleep "$DELAY"

for N in "${NODES[@]}"; do
    PID="$(pgrep -f "$N" | head -1)"
    if [ -z "$PID" ]; then
        echo "" >> "$OUT"; echo "----- [$N] 실행 중 아님 -----" >> "$OUT"
        continue
    fi
    (
        RAW="$(mktemp)"
        py-spy record --pid "$PID" --duration "$DUR" --format raw -F \
                      "${NONBLOCK[@]}" -o "$RAW" 2>/dev/null
        TOTAL="$(awk '{s+=$NF} END{print s+0}' "$RAW")"
        {
            echo ""
            if [ "${TOTAL:-0}" -eq 0 ]; then
                echo "----- [$N] pid=$PID  표본 0개 (붙기 실패했거나 순수 C 구간) -----"
            else
                echo "----- [$N] pid=$PID  표본=$TOTAL -----"
                # 스택 맨 끝 프레임(실제로 실행 중이던 함수)으로 집계
                awk -v tot="$TOTAL" '{
                    n=$NF; $NF=""
                    m=split($0, a, ";"); leaf=a[m]
                    gsub(/^ +| +$/, "", leaf)
                    c[leaf]+=n
                } END {
                    for (f in c) printf "%6.2f%%  %8d  %s\n", 100*c[f]/tot, c[f], f
                }' "$RAW" | sort -rn | head -25
            fi
        } >> "$OUT"
        rm -f "$RAW"
    ) &
done

wait
echo "" >> "$OUT"
echo "# cpu_profile 완료 $(date '+%F %T')" >> "$OUT"
exit 0
