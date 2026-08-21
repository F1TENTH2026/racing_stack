# Racing stack architecture audit (local `kyumin`, 2026-08-21)

This note describes the checked-out source, not an older GitHub revision. Rates
marked "configured" are timer/config targets; they are not laptop or Jetson
measurements. The default rclpy entry points use a single-threaded executor unless
an external executor is supplied.

## Pipeline and cost audit

| Area / node | Lang | Main input -> output | Rate / QoS | Executor / blocking and optional work | Cost / criticality |
|---|---|---|---|---|---|
| `urg_node_driver` | C++ | Hokuyo -> `/scan` | device-driven, `skip=0`; actual rate requires measurement | async driver; intensity enabled | MEDIUM / critical |
| VESC driver + ackermann conversion | C++ | serial, commands -> state, `/vesc/odom`, actuator | device/command rate; bounded ROS queues | serial I/O outside controller | MEDIUM / safety critical |
| VESC IMU | C++ | VESC -> `/vesc/sensors/imu` | polled by a 100 Hz timer; output is response-driven | gyro is measured; `/vesc/odom.angular.z` is servo-model-derived | LOW / critical input |
| local EKF | C++ (`robot_localization`) | VESC `vx` + IMU gyro `wz` -> `/odometry/local`, `odom->base_link` | configured 100 Hz, queue 1 | separate process | MEDIUM / critical |
| particle filter | Python + `range_libc` C/C++ | `/scan`, VESC `vx`, IMU `wz`, map -> `/pf/pose/odom` | scan-driven; sensor QoS KEEP_LAST/BEST_EFFORT; state depth 1 | rclpy single-threaded; startup map service wait; ray cast/sensor model in C backend; NumPy resampling/motion/covariance | HIGH / critical |
| global EKF | C++ (`robot_localization`) | local twist + PF absolute pose -> `/car_state/odom`, `map->odom` | configured 100 Hz, queue 1 | separate process | MEDIUM / critical |
| Frenet odom | Python/NumPy | `/car_state/odom`, raceline -> `/car_state/odom_frenet` | input-driven, depth 10 | conversion work per odom | MEDIUM / critical |
| detect | C++ | latest `/scan`, odom/raceline -> `/detect/raw_obstacles` | independent 40 Hz wall timer; scan SensorDataQoS | may reuse a scan; clustering/L-shape loops; marker messages every timer | HIGH in obstacle modes / critical |
| tracking | Python/NumPy | raw obstacles -> `/tracking/obstacles[_raw]` | configured timer/input path; depth from params | object association and filtering; markers | HIGH in obstacle modes / critical |
| state machine | Python/NumPy/SciPy | tracking, paths, odom -> `/behavior_strategy` and state | configured 40 Hz (was 80 Hz), mostly depth 10 | single-thread timer; transition TTL/hysteresis are elapsed-seconds based; optional disk debug logs | HIGH in obstacle modes / critical |
| static avoidance | Python/NumPy/SciPy | tracked static objects + bounds -> static avoidance path | input/timer driven, depth 10 | optional debug file output; spline candidates | HIGH in Q2/H2H / critical |
| dynamic lane/SQP planners | Python/NumPy/SciPy | opponent/state -> avoidance paths | input driven | lane visualizer can block on matplotlib but launch default is off | HIGH in H2H / critical |
| GP prediction chain | Python + NumPy/sklearn | opponent history -> prediction | event/service driven | service waits/spinning exist; former launch hardcoded `taskset -c 1`; matplotlib imported by predictor module | HIGH in H2H / non-actuator-critical |
| controller manager + L1 | Python/NumPy | behavior, global/local odom, optional scan -> Ackermann | timer 50 Hz; subscriptions depth 10, scan SensorDataQoS | single-threaded; marker construction now subscriber-gated; FTG scan subscription mode-gated | HIGH / safety critical |
| simple mux | Python | autonomy/joy/estop -> `/vesc/ackermann_cmd` | callback driven | no laptop response required | MEDIUM / safety critical |
| VESC command bridge | C++ | Ackermann -> motor/servo | command driven | braking logic and odom timeout in driver config | MEDIUM / safety critical |
| lap/pitwall/visualization | Python/C++ | state topics -> metrics/markers/network viewers | varied | optional, not in actuator dependency chain; lap analyser is still launched by base | LOW-MEDIUM / noncritical |

## TF ownership (PF/cartographer real-car localization)

| Edge | Sole owner | Explicitly disabled competitors |
|---|---|---|
| `map -> odom` | global `ekf_localization` | PF TF false; cartographer `publish_to_tf=false`; VESC has no map TF |
| `odom -> base_link` | `ekf_local_odom` | `vesc_to_odom publish_tf=false`; global EKF emits only its REP-105 map correction |
| `base_link -> laser` | `static_tf_publisher` from `CAR/static_transforms.yaml` | PF TF false |

KISS is an intentional alternative topology (`map -> base_link` owned by KISS),
not started together with either EKF. Simulator TF is owned by the gym bridge.

## Before and identified hazards

```text
VESC vx + servo-command bicycle wz --+--> global EKF --> map->odom
VESC odom node ----------------------------> odom->base_link
LiDAR --> PF (latest speed/latest gyro * scan dt) --> PF pose stamped now()
LiDAR --> detect --> Python tracking --> state/planners --> Python controller
                                           GP always on, fixed CPU core
```

The high-consequence findings were: PF timestamp laundering, unsynchronised PF
motion input, modeled yaw rate fused as a measurement by the global EKF,
callback-count trajectory freshness checked after control, non-finite speed
holding the previous high-speed command, stale-path steering reset to zero,
empty curvature slices, ineffective L1 validation, nominal rather than elapsed
PID `dt`, unconditional controller scan deserialization, and an always-on GP chain.

## Mode pipelines after this phase

```text
TIME_TRIAL: sensors -> dual localization -> scaled raceline adapter
           -> controller -> mux -> VESC

STATIC_QUALIFY: sensors -> dual localization -> detect -> tracking
                -> state + static/recovery/start planning
                -> controller -> mux -> VESC

HEAD_TO_HEAD/PRACTICE: STATIC pipeline + dynamic planner + GP prediction
                       -> controller -> mux -> VESC
```

`race_mode` is a launch-time topology choice; no node exposes it as a mutable
runtime parameter. Practice retains debug/visualization opt-ins. The state machine
was reduced from 80 Hz to the 40 Hz perception cadence: with a phase-independent
40 Hz producer, 40 of 80 timer cycles per second (50%) could not contain a new
tracking sample. Event-driven execution was not selected because odometry and
planner inputs have different rates and the controller benefits from a bounded
publication cadence. GPU PF and C++ controller/tracker migration remain gated on
profiling and parity evidence; the 100 Hz C++ EKFs remain unchanged until profiling
shows that they are a bottleneck.

## Rate decision and migration priorities

The 40 Hz state-machine timer is deliberately periodic rather than directly invoked
by `/tracking/obstacles`: tracking is the decision freshness boundary, but odometry,
recovery and avoidance paths arrive independently. A periodic 25 ms snapshot gives
those inputs a deterministic merge point and continues publishing a bounded-cadence
path to the 50 Hz controller. All active rate-sensitive holds in the state machine now use
ROS elapsed seconds: overtaking TTL, static overtaking TTL, FTG low-speed delay,
planner message/cache TTLs, planner hysteresis/killing timers, and the post-overtake
recovery grace. Changing the timer rate therefore does not change their durations.
The legacy `splini_ttl`, `pred_splini_ttl`, and `splini_hyst_timer_sec` compatibility
parameters are retained but are not referenced by the current transition/path logic;
they therefore do not hide a cycle-based duration.

The controller remains at 50 Hz. Its Python manager and combined controller are a
good C++ migration target because they are actuator-critical and execute NumPy path
selection, curvature and control logic every 20 ms. A port should be accepted only
after bag replay shows equivalent source selection, finite-command safety behavior,
steering/speed output tolerances, and stale-input braking.

Tracking remains at 40 Hz. Its Python association loop, Kalman predict/update,
static/dynamic voting and marker construction are the next C++ migration target.
Replay parity must cover ID continuity, association gates, TTL expiry, classification
latency and `/tracking/obstacles` ordering/stamps; changing languages without those
checks would alter state-machine behavior even at the same nominal rate.

PF remains scan-driven at the LiDAR's 40 Hz freshness boundary. The race profile uses
the `pcddt` C++ range backend, SensorDataQoS for scan/IMU, depth 1 for odometry, a
25 ms deadline profiler, and disables the optional fake-scan/full-cloud visualization
work by default. The two 100 Hz EKFs are already C++ (`robot_localization`) and remain
unchanged because this audit contains no measured evidence that they miss deadlines
or dominate CPU.
