#!/usr/bin/env bash
# leo.sh — enter the LEO test workspace, isolated from the main (thirdimpact) one.
#
# SOURCE it (do not execute). Works in bash AND zsh. Add the alias once, pointing
# at THIS checkout — not the roboracer_ws one:
#     alias leo='source /home/user/leo_ws/src/racing_stack/leo.sh'
# then just run:  leo
#
# WHY THIS FILE EXISTS
# --------------------
# The car keeps two checkouts of the same repo, in two colcon workspaces, with the
# SAME folder name (<ws>/src/racing_stack):
#
#     /home/user/roboracer_ws/src/racing_stack   branch main   -> `thirdimpact`
#     /home/user/leo_ws/src/racing_stack         branch leo    -> `leo`   (this)
#
# The folder name is NOT what selects the code — the *entry script you source* is.
# thirdimpact.sh derives everything from its own $BASH_SOURCE (_URS_REPO, then
# _URS_WS=<repo>/../..), so sourcing the copy that lives inside THIS checkout
# resolves the repo, the colcon overlay, cyclonedds.xml and RAYCASTER_DIR to this
# workspace. leo.sh therefore does not duplicate that logic; it sets three knobs
# and hands off to its OWN sibling thirdimpact.sh.
#
# WHAT IS ISOLATED, AND WHAT IS NOT
#   isolated: conda env, colcon overlay (install/), ROS domain, DDS config, build/
#   shared:   the machine, the hardware (VESC/LiDAR are physical, one at a time)
# Two stacks must never drive the car at once — separate domains stop them seeing
# each other's TOPICS, not from fighting over the same /dev/ttyACM*.

# --- resolve this script's own path (bash sets BASH_SOURCE; zsh uses ${(%):-%x}) ---
if [ -n "${BASH_SOURCE:-}" ]; then
    _LEO_SRC="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _LEO_SRC="${(%):-%x}"
else
    _LEO_SRC="$0"
fi
_LEO_REPO="$(cd "$(dirname "$_LEO_SRC")" && pwd)"

# --- 1) dedicated conda env ---
# Cloned from thirdimpact:  conda create --clone thirdimpact -n leo
# A separate env (rather than sharing thirdimpact) matters for ONE reason: the
# f110_gym editable install. thirdimpact.sh re-points it at the repo it was
# sourced from, so a shared env would make the last-sourced workspace win and
# silently break `gym_bridge` in the other one. Two envs, two editable installs,
# no thrash.
URS_CONDA_ENV="${LEO_CONDA_ENV:-leo}"
if ! conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$URS_CONDA_ENV"; then
    if [ -z "$(command -v conda)" ] || [ ! -d "$(conda info --base 2>/dev/null)/envs/$URS_CONDA_ENV" ]; then
        echo "ERROR: conda env '$URS_CONDA_ENV' not found. Create it with:" >&2
        echo "    conda create --clone thirdimpact -n $URS_CONDA_ENV -y" >&2
        return 1 2>/dev/null || exit 1
    fi
fi
export URS_CONDA_ENV
export URS_ENV_LABEL=leo

# --- 2) its own ROS domain ---
# thirdimpact.sh does `ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"`, i.e. it KEEPS a value
# already in the environment. So setting it here (unconditionally, since the shell
# may still carry 1 from a previous `thirdimpact`) is what actually decides it.
# Domain 1 = main stack, domain 2 = leo. Different domains do not exchange DDS
# discovery traffic, so the two stacks cannot cross-wire each other's topics.
export ROS_DOMAIN_ID="${LEO_ROS_DOMAIN_ID:-2}"

# --- 3) hand off to THIS checkout's thirdimpact.sh ---
# Everything below (conda activate, ROS-leak filtering, CycloneDDS, the colcon
# overlay, cbuild/ros2kill) is that script's job, unchanged. The path is anchored
# to $_LEO_REPO, so it is impossible for this to pick up the other workspace's copy.
if [ ! -f "$_LEO_REPO/thirdimpact.sh" ]; then
    echo "ERROR: $_LEO_REPO/thirdimpact.sh not found — is this a full racing_stack checkout?" >&2
    return 1 2>/dev/null || exit 1
fi
source "$_LEO_REPO/thirdimpact.sh"

# --- 4) say out loud which code is live ---
# The whole point of the second workspace is running DIFFERENT code, so print the
# branch and the commit rather than making you run `git status` to be sure.
_leo_branch="$(git -C "$_LEO_REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
_leo_sha="$(git -C "$_LEO_REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[leo] repo=$_LEO_REPO"
echo "[leo] branch=$_leo_branch ($_leo_sha)"
if [ "$_leo_branch" != "leo" ]; then
    echo "[leo] WARNING: this checkout is on '$_leo_branch', not 'leo'." >&2
fi
unset _leo_branch _leo_sha _LEO_SRC
