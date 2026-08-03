#!/bin/bash
# Accuracy comparison: all backends' scans, color-coded, same pose, in RViz.
#   ./run_compare.sh [map]        e.g.  ./run_compare.sh f
# Toggle backends on/off in the RViz Displays panel; edit CONFIGS in compare_demo.py
# to add backends / theta_disc values. Set -p ego_speed:=20 to drive the pose.
set -e
source /opt/ros/jazzy/setup.bash
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -vi anaconda | paste -sd:)
RC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$RC:$RC/range_libc/pywrapper:$PYTHONPATH"
export ROS_DOMAIN_ID=1
MAP=${1:-f}
echo "compare demo: every backend color-coded (green rm=ref, red glt112, yellow glt720, blue cddt112, white bl)"
/usr/bin/python3 "$RC/examples/compare_demo.py" --ros-args -p map:="$MAP" &
NODE=$!; trap "kill $NODE 2>/dev/null" EXIT
rviz2 -d "$RC/examples/compare_demo.rviz"
