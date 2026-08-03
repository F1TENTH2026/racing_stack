#!/bin/bash
###############################################################################
# build_packages_on_car.sh
#
# Build the UNICORN racing-stack ROS 2 Jazzy workspace (Ubuntu 24.04 + Jazzy).
#
# Architecture: this meta-repo (<ws>/src/unicorn-racing-stack) is SELF-CONTAINED
# — all ported/migrated packages live under it. The sibling component repos
# (creating_autonomous_car, race_stack, *-ros1) under <ws>/src stay
# COLCON_IGNORE'd; CAC is used only for the pip-editable f1tenth_gym sim lib and
# the range_libc source. cartographer is installed from apt (NOT built).
#
# Usage: bash build_packages_on_car.sh
###############################################################################
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this meta-repo
WS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"                  # colcon workspace root
CAC="${WS_DIR}/src/creating_autonomous_car"
SUDO="sudo"; [ "$(id -u)" = "0" ] && SUDO=""                # no sudo when root (Docker)

echo "============================================"
echo " Building UNICORN racing-stack (Jazzy)"
echo " Workspace : ${WS_DIR}"
echo "============================================"

# 1. apt dependencies (cartographer + serial + ackermann + sklearn/skimage from apt) ----
echo "[1/7] Installing apt dependencies..."
$SUDO apt-get update
$SUDO apt-get install -y \
    python3-rosdep python3-pip python3-vcstool python3-colcon-common-extensions \
    cython3 python3-numpy python3-scipy python3-skimage python3-sklearn python3-tqdm \
    python3-matplotlib python3-opencv \
    ros-jazzy-xacro ros-jazzy-rmw-cyclonedds-cpp ros-jazzy-ackermann-msgs \
    ros-jazzy-serial-driver ros-jazzy-io-context \
    ros-jazzy-cartographer ros-jazzy-cartographer-ros \
    libboost-dev libboost-iostreams-dev libpcl-dev libyaml-cpp-dev google-mock

# 2. Import sibling component repos (idempotent; skips existing dirs) ----------
echo "[2/7] Importing component repos via vcstool..."
mkdir -p "${WS_DIR}/src"
vcs import "${WS_DIR}/src" < "${SCRIPT_DIR}/unicorn.repos" || true
# nested Hokuyo driver repos for the urg_node copy (hardware)
if [ -f "${SCRIPT_DIR}/sensor/urg_node/additional_repos.repos" ]; then
    vcs import "${SCRIPT_DIR}/sensor/urg_node" < "${SCRIPT_DIR}/sensor/urg_node/additional_repos.repos" || true
fi

# 3. Python deps — PIN numpy<2 (ROS Jazzy ABI); planner/optimizer libs ---------
echo "[3/7] Installing Python dependencies (numpy<2 pinned)..."
PIPF="--break-system-packages"
echo "numpy<2" > /tmp/unicorn_constraints.txt
pip3 install $PIPF "numpy<2"
pip3 install $PIPF -c /tmp/unicorn_constraints.txt \
    transforms3d pynput tqdm casadi \
    trajectory_planning_helpers \
    "git+https://github.com/ForzaETH/CCMA.git"
# quadprog must be compiled against the pinned numpy (no build isolation)
pip3 install $PIPF -c /tmp/unicorn_constraints.txt --no-binary :all: --no-build-isolation quadprog
# f1tenth_gym sim (editable, from CAC)
[ -d "${CAC}/simulator/f1tenth_gym" ] && pip3 install $PIPF -e "${CAC}/simulator/f1tenth_gym"

# 4. range_libc (modernized, from the raycaster submodule) --------------------
echo "[4/7] Installing range_libc..."
if [ -d "${SCRIPT_DIR}/tools/raycaster/range_libc/pywrapper" ]; then
    ( cd "${SCRIPT_DIR}/tools/raycaster/range_libc/pywrapper" && pip3 install . $PIPF ) || \
        echo "  WARNING: range_libc build failed (localization only)."
fi

# 5. rosdep -------------------------------------------------------------------
echo "[5/7] Running rosdep..."
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || $SUDO rosdep init || true
rosdep update || true
rosdep install --from-paths "${SCRIPT_DIR}" --ignore-src -r -y || true

# 6. Build (CAC etc. stay COLCON_IGNORE'd; only unicorn-racing-stack builds) ---
echo "[6/7] Building workspace with colcon..."
cd "${WS_DIR}"
source /opt/ros/jazzy/setup.bash
export PYTHONNOUSERSITE=0
colcon build --symlink-install \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release

# 7. ~/.bashrc (idempotent) ---------------------------------------------------
echo "[7/7] Updating ~/.bashrc..."
BASHRC="${HOME}/.bashrc"
grep -qF "source ${WS_DIR}/install/setup.bash" "${BASHRC}" 2>/dev/null || \
    printf '\n# UNICORN racing-stack (Jazzy)\nsource %s/install/setup.bash\n' "${WS_DIR}" >> "${BASHRC}"
grep -qF "RMW_IMPLEMENTATION" "${BASHRC}" 2>/dev/null || \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> "${BASHRC}"

echo ""
echo "============================================"
echo " Build complete!"
echo "============================================"
