# pitwall

Ergonomic, one-line telemetry logging for the UNICORN racing stack — record only
what you want, from anywhere in any node, into a single [MCAP](https://mcap.dev)
file you open directly in Foxglove. No per-node files, no message definitions, no
publisher boilerplate at the call site.

## Idea

```cpp
pitwall::log("speed", 2.0);     // C++ — anywhere, any node
```
```python
pitwall.log("speed", 2.0)       # Python — same API
```

Under the hood `log(key, value)` lazily creates a publisher on `/pitwall/<key>`
(`std_msgs/Float64`) and publishes **only when a recorder is subscribed** —
otherwise it is a cheap no-op (`get_subscription_count() == 0` gate). So:

- **No per-node files.** Data flows over a topic, not to disk in each node.
- **Recorder alive → recorded; recorder absent → nothing happens.** Standard
  pub/sub lazy-consumer semantics.
- **Typed channels.** `Float64` per key → Foxglove plots each key out of the box.
- **One MCAP.** The recorder captures every `/pitwall/*` (plus sensor topics)
  into a single file — no merge step.
- **Live or offline, same tool.** The topics are normal visible topics, so
  foxglove_bridge streams them live to Foxglove/Lichtblick, and the recorded
  MCAP opens in the same app offline.
- **Hideable.** Set env `PITWALL_TOPIC_PREFIX=/_pitwall` (leading underscore) to
  make them ROS *hidden topics* (out of `ros2 topic list`); the recorder passes
  `--include-hidden-topics` so capture still works.

## Usage

### 1. Make pitwall available

pitwall is a normal package in this colcon workspace. There is **no separate
install step** — building the workspace puts it under `install/`, and sourcing
that overlay makes it a discoverable ROS dependency for every other package:

```bash
cd /home/js/unicorn_racing_stack
colcon build --symlink-install            # builds pitwall (and the rest)
source install/setup.bash                 # now `pitwall` is findable
```

After that, other packages declare a dependency on it and just use it — no
copying headers, no manual linking paths.

### 2. Use it from a C++ node

**`package.xml`** — add the dependency:
```xml
<depend>pitwall</depend>
```

**`CMakeLists.txt`** — find it and link your target (pulls in headers + lib):
```cmake
find_package(pitwall REQUIRED)
# ...
add_executable(my_node src/my_node.cpp)
ament_target_dependencies(my_node rclcpp pitwall)   # <-- pitwall here
```

**code** — `#include <pitwall/pitwall.hpp>`:
```cpp
#include <pitwall/pitwall.hpp>

class MyNode : public rclcpp::Node {
public:
  MyNode() : Node("my_node") {
    pitwall::init(this);                       // once; reuses this node's DDS participant
    timer_ = create_wall_timer(20ms, [this]{
      pitwall::log("speed", speed_);           // -> /pitwall/speed (Float64)
      pitwall::log("state.x", x_);             // dots sanitized: /pitwall/state_x
      pitwall::event("lap_start");             // sparse event on /pitwall/events
    });
  }
};
```

### 3. Use it from a Python node

**`package.xml`** — Python import dependency:
```xml
<exec_depend>pitwall</exec_depend>
```

**code** — `import pitwall` (nothing to compile/link):
```python
import pitwall

class MyNode(Node):
    def __init__(self):
        super().__init__("my_node")
        pitwall.init(self)                     # once, with your rclpy node
        self.create_timer(0.02, self._tick)

    def _tick(self):
        pitwall.log("speed", self.speed)       # -> /pitwall/speed
        pitwall.event("lap_start")
```

`init()` is optional but recommended — without it the first `log()` lazily spawns
a hidden node (one extra DDS participant per process). C++ and Python nodes share
the same `/pitwall/*` topics, so a single recorder captures both.

### 4. Record

```bash
ros2 launch pitwall record.launch.py output_dir:=~/runs/lap03
# add sensors to the same MCAP via the regex:
ros2 launch pitwall record.launch.py \
    output_dir:=~/runs/lap03 \
    topic_regex:='/pitwall/.*|/scan|/imu|/camera/.*'
```

Ctrl-C → launch sends SIGINT → `ros2 bag record` finalizes the MCAP cleanly.
A `pitwall_monitor` node samples the recorder process's CPU%/RSS and the system
load and logs them on `/pitwall/monitor/*` into the same file, so you can see
recording cost on the same Foxglove timeline.

### 5. View live (Foxglove / Lichtblick over foxglove_bridge)

Run the bridge on the machine that has the data (e.g. the car), then connect the
viewer from your laptop:

```bash
sudo apt install ros-jazzy-foxglove-bridge      # once
ros2 launch pitwall live.launch.py              # bridge only (ws://<host>:8765)
#   or record AND stream at the same time:
ros2 launch pitwall record.launch.py live:=true output_dir:=~/runs/lap03
```

In Foxglove/Lichtblick: **Open connection → `ws://<host-ip>:8765`**, then add a
Plot panel and the `/pitwall/*` series. Only topics you actually display are
streamed (the bridge subscribes lazily), so live bandwidth stays bounded — keep
heavy sensors (camera/lidar) in the on-car recording and stream them live only
when needed.

### 6. View a recorded MCAP (offline)

No bridge needed — open the file directly:

```bash
~/tools/lichtblick/lichtblick --no-sandbox ~/runs/lap03/lap03_0.mcap
# or in Foxglove/Lichtblick: Open local file
mcap info ~/runs/lap03/lap03_0.mcap            # quick headless check
```

## Try it

```bash
# terminal 1 — recorder
ros2 launch pitwall record.launch.py output_dir:=/tmp/run

# terminal 2 — demo producer (C++ or Python)
ros2 run pitwall pitwall_demo
#   or: ros2 run pitwall demo_producer.py

# Ctrl-C both, then inspect:
mcap info /tmp/run/run_0.mcap
```

## Build notes (this workspace)

- ROS 2 **Jazzy**. Build from the workspace root with `colcon build --symlink-install`.
- **Anaconda conflict:** the host's `python3` is Anaconda, whose `libstdc++`
  lacks `GLIBCXX_3.4.30` and breaks ROS Python extensions. Build/run with system
  python on PATH (`export PATH=/usr/bin:$PATH`) or a conda-free shell; the build
  here pins `-DPython3_EXECUTABLE=/usr/bin/python3`.
- Scripts in `scripts/` must stay executable (`chmod +x`) — with
  `--symlink-install` the install symlink points at the source file.

## Layout

```
pitwall/
├── include/pitwall/pitwall.hpp   C++ API
├── src/pitwall.cpp               C++ producer library
├── src/demo_producer.cpp         C++ demo (pitwall_demo)
├── pitwall/__init__.py           Python API
├── scripts/demo_producer.py      Python demo
├── scripts/monitor_node.py       recorder CPU/RSS monitor -> /pitwall/monitor/*
├── launch/record.launch.py       ros2 bag record (mcap) + monitor (+ live:=true)
├── launch/live.launch.py         foxglove_bridge only (live viewing)
└── config/topics.yaml            reference sensor-topic list
```
