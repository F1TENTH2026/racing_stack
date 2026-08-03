# unicorn-gym

ROS 2 (Jazzy) F1TENTH simulation packages, **based on
[f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) and
[f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros)**, extended for the
UNICORN racing stack (virtual opponent / obstacle injection, `range_libc`-backed
scans, an RViz sim control panel).

## Contents

- `f1tenth_gym/`       — physics core (`f110_gym`), based on f1tenth_gym
- `f1tenth_gym_ros/`   — ROS 2 bridge, based on f1tenth_gym_ros

> The RViz Sim Control panel (inject / clear obstacles, drive the opponent, plus
> the state banner + telemetry feed) now lives in the **`pitwall`** package
> (`race_utils/pitwall`), registered as the `pitwall/SimControlPanel` RViz panel.

## Raycaster (2D-RayCaster)

The fast scan backend uses
[2D-RayCaster](https://github.com/jeongsang-ryu/2D-RayCaster). Clone it next to
this package and point `RAYCASTER_DIR` at it:

```bash
git clone https://github.com/jeongsang-ryu/2D-RayCaster.git
export RAYCASTER_DIR=$PWD/2D-RayCaster
# optional fast C++ backends (rm / cddt / glt); the default `lut` needs nothing:
pip install --no-build-isolation -e 2D-RayCaster/range_libc/pywrapper
```

> Inside the `unicorn-racing-stack` workspace this is already provided as the
> `race_utils/raycaster` submodule, and `unicorn.sh` sets `RAYCASTER_DIR` for you.

## Usage

Built with `colcon` as part of a ROS 2 workspace. `f1tenth_gym` is a pip-editable
package (`pip install --no-build-isolation -e f1tenth_gym`); `f1tenth_gym_ros` is
an ament/colcon package. See the parent
[unicorn-racing-stack](https://github.com/hmcl-unist/unicorn-racing-stack)
`README.md` ("Get started") for the full environment setup.
