#!/bin/bash

ot_dir="$(ros2 pkg prefix overtaking_sector_tuner)"
install_dir="$(dirname "$ot_dir")"
ws_dir="$(dirname "$install_dir")"
# Move to ws directory or abort if directory doesn't exist
cd "$ws_dir" || exit

colcon build --packages-select stack_master
