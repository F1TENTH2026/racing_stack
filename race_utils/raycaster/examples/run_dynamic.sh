#!/bin/bash
# Dynamic-overlay demo: precompute static (glt/lut) + overlay opponent & clicked obstacles.
#   ./run_dynamic.sh [backend] [map]     e.g.  ./run_dynamic.sh glt f
# In RViz use the "Publish Point" tool to add/remove obstacles (they appear in /scan).
set -e
source /opt/ros/jazzy/setup.bash
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -vi anaconda | paste -sd:)
RC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$RC:$RC/range_libc/pywrapper:$PYTHONPATH"
export ROS_DOMAIN_ID=1
BACKEND=${1:-glt}; MAP=${2:-f}
echo "dynamic demo: backend=$BACKEND map=$MAP (static precompute + dynamic overlay)"
/usr/bin/python3 "$RC/examples/dynamic_demo.py" --ros-args -p backend:="$BACKEND" -p map:="$MAP" &
NODE=$!; trap "kill $NODE 2>/dev/null" EXIT
rviz2 -d "$RC/examples/dynamic_demo.rviz"
