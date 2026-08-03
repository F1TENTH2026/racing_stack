#!/bin/bash
# Launch the RaycastEngine RViz demo (map + live scan).
#   ./run_demo.sh [backend] [map]      e.g.  ./run_demo.sh rm f   |   ./run_demo.sh pcddt test
# Needs: ROS 2 Jazzy, and range_libc built for the system python:
#   (cd ../range_libc/pywrapper && WITH_CUDA=OFF /usr/bin/python3 setup.py build_ext --inplace)
set -e
source /opt/ros/jazzy/setup.bash
# use the ROS python (3.12), not anaconda
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -vi anaconda | paste -sd:)
RC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$RC:$RC/range_libc/pywrapper:$PYTHONPATH"
export ROS_DOMAIN_ID=1
BACKEND=${1:-rm}; MAP=${2:-f}
echo "RaycastEngine demo: backend=$BACKEND map=$MAP"
/usr/bin/python3 "$RC/examples/rviz_demo.py" --ros-args -p backend:="$BACKEND" -p map:="$MAP" &
NODE=$!
trap "kill $NODE 2>/dev/null" EXIT
rviz2 -d "$RC/examples/raycast_demo.rviz"
