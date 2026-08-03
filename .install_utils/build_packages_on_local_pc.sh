#!/bin/bash
###############################################################################
# build_packages_on_local_pc.sh
#
# Build the UNICORN racing-stack ROS 2 Jazzy workspace on a local PC
# (Ubuntu 24.04 + ROS 2 Jazzy) — SIMULATION profile.
#
# This meta-repo lives at <workspace>/src/unicorn-racing-stack and orchestrates
# the workspace via vcstool (unicorn.repos). Component repos are imported as
# SIBLINGS into <workspace>/src (e.g. creating_autonomous_car). Car-only
# packages are excluded by the COLCON_IGNORE markers shipped inside CAC.
#
# Usage: bash build_packages_on_local_pc.sh
###############################################################################
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this meta-repo
WS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"                  # colcon workspace root
CAC="${WS_DIR}/src/creating_autonomous_car"

echo "============================================"
echo " Building UNICORN racing-stack (LOCAL PC / sim)"
echo " Workspace : ${WS_DIR}"
echo " Meta-repo : ${SCRIPT_DIR}"
echo "============================================"

# 1. apt dependencies + workspace tooling -------------------------------------
echo "[1/6] Installing apt dependencies..."
sudo apt update
sudo apt install -y \
    python3-rosdep python3-pip python3-skimage \
    python3-vcstool python3-colcon-common-extensions \
    ros-jazzy-xacro ros-jazzy-rmw-cyclonedds-cpp gedit

# 2. Import component repos into the workspace src/ (siblings) -----------------
echo "[2/6] Importing component repos via vcstool..."
mkdir -p "${WS_DIR}/src"
# Clones any MISSING component into ${WS_DIR}/src; existing checkouts (e.g. a
# creating_autonomous_car you actively develop) are left untouched. To update
# tracked repos in place:  vcs pull "${WS_DIR}/src"
vcs import "${WS_DIR}/src" < "${SCRIPT_DIR}/unicorn.repos"

# 3. Python dependencies (f1tenth_gym sim + helpers, not covered by rosdep) ----
echo "[3/6] Installing Python dependencies..."
pip install -e "${CAC}/simulator/f1tenth_gym" --break-system-packages
pip install transforms3d pynput --break-system-packages
pip install --upgrade coverage --break-system-packages

# 4. rosdep -------------------------------------------------------------------
echo "[4/6] Running rosdep..."
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update
rosdep install --from-paths "${WS_DIR}/src" --ignore-src -r -y

# 5. Build --------------------------------------------------------------------
echo "[5/6] Building workspace with colcon..."
cd "${WS_DIR}"
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# 6. ~/.bashrc (idempotent) ---------------------------------------------------
echo "[6/6] Updating ~/.bashrc..."
BASHRC="${HOME}/.bashrc"
SOURCE_LINE="source ${WS_DIR}/install/setup.bash"
if ! grep -qF "${SOURCE_LINE}" "${BASHRC}" 2>/dev/null; then
    printf '\n# UNICORN racing-stack (Jazzy) workspace\n%s\n' "${SOURCE_LINE}" >> "${BASHRC}"
fi
if ! grep -qF "RMW_IMPLEMENTATION" "${BASHRC}" 2>/dev/null; then
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> "${BASHRC}"
fi
if ! grep -qF "alias cb=" "${BASHRC}" 2>/dev/null; then
    echo "alias cb='cd ${WS_DIR} && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> "${BASHRC}"
    echo "alias sauce='source ${WS_DIR}/install/setup.bash'" >> "${BASHRC}"
fi

echo ""
echo "============================================"
echo " Build complete! (LOCAL PC / sim)"
echo " Run 'source ~/.bashrc' or open a new terminal."
echo "============================================"
