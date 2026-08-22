#!/usr/bin/env python3
"""check_scaling.py — is the map's speed scaling actually reaching the car?

Compares the raceline as generated (/global_waypoints) against what
speed_sector_tuner publishes (/global_waypoints_scaled) and prints the ratio
per sector, so "the car did not speed up after the phase switch" becomes a
number instead of a feeling.

  ros2 run stack_master check_scaling.py

Reads three things and lines them up:
  * /global_waypoints          -- vx_mps as the optimiser produced it
  * /global_waypoints_scaled   -- what the state machine and controller consume
  * /speed_sector_tuner's params -- use_sector_scaling, default_scaling,
                                    phase_multiplier and the sector table

Expected: with use_sector_scaling True the ratio in each sector equals that
sector's `scaling` x `phase_multiplier`; with it False every ratio equals
default_scaling. A ratio that matches default_scaling while the parameter says
True means the tuner never re-scaled -- report that, it is a real bug.

`ros2 topic echo --field wpnts[0].vx_mps` does NOT work for this: the field
selector has no array indexing, which is why this script exists.
"""

import sys
import time

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from f110_msgs.msg import WpntArray

TIMEOUT = 15.0


class CheckScaling(Node):
    def __init__(self):
        super().__init__('check_scaling')
        self.og = None
        self.sc = None
        self.create_subscription(WpntArray, '/global_waypoints', self._og_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints_scaled', self._sc_cb, 10)

    def _og_cb(self, msg):
        self.og = msg

    def _sc_cb(self, msg):
        self.sc = msg

    def wait(self):
        t0 = time.time()
        while rclpy.ok() and (self.og is None or self.sc is None):
            if time.time() - t0 > TIMEOUT:
                missing = [n for n, v in (('/global_waypoints', self.og),
                                          ('/global_waypoints_scaled', self.sc)) if v is None]
                self.get_logger().error(
                    f'timed out waiting for {", ".join(missing)} — is the stack running?')
                return False
            rclpy.spin_once(self, timeout_sec=0.2)
        return True

    def get_params(self, names):
        cli = self.create_client(GetParameters, '/speed_sector_tuner/get_parameters')
        if not cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/speed_sector_tuner/get_parameters unavailable')
            return {}
        req = GetParameters.Request()
        req.names = list(names)
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if fut.result() is None:
            return {}
        out = {}
        for name, v in zip(names, fut.result().values):
            # ParameterType: 1 bool, 2 int, 3 double, 4 string
            out[name] = {1: v.bool_value, 2: v.integer_value,
                         3: v.double_value, 4: v.string_value}.get(v.type)
        return out


def main():
    rclpy.init()
    node = CheckScaling()
    if not node.wait():
        node.destroy_node(); rclpy.shutdown(); sys.exit(1)

    base = node.get_params(['use_sector_scaling', 'default_scaling',
                            'phase_multiplier', 'n_sectors'])
    use_sec = base.get('use_sector_scaling')
    n_sec = int(base.get('n_sectors') or 0)
    phase = base.get('phase_multiplier')
    dflt = base.get('default_scaling')

    print()
    print(f'  use_sector_scaling : {use_sec}')
    print(f'  default_scaling    : {dflt}      (섹터 OFF일 때 전 구간에 적용)')
    print(f'  phase_multiplier   : {phase}      (섹터 ON일 때 각 섹터에 곱해짐)')
    print(f'  n_sectors          : {n_sec}')

    og, sc = node.og, node.sc
    if len(og.wpnts) != len(sc.wpnts):
        print(f'\n  ! 길이 불일치: 원본 {len(og.wpnts)} vs 스케일 {len(sc.wpnts)}')

    n = min(len(og.wpnts), len(sc.wpnts))
    ratios = [(sc.wpnts[i].vx_mps / og.wpnts[i].vx_mps) if og.wpnts[i].vx_mps > 1e-6 else float('nan')
              for i in range(n)]

    sectors = []
    for i in range(n_sec):
        p = node.get_params([f'Sector{i}.start', f'Sector{i}.end', f'Sector{i}.scaling'])
        sectors.append((i, int(p.get(f'Sector{i}.start') or 0),
                        int(p.get(f'Sector{i}.end') or 0),
                        p.get(f'Sector{i}.scaling')))

    print(f'\n  웨이포인트 {n}개 — 실측 배율 (스케일 vx / 원본 vx)')
    print(f'  {"섹터":<8}{"범위":>14}{"yaml scaling":>14}{"기대":>8}{"실측(중앙)":>12}{"":>4}')
    print('  ' + '-' * 62)
    ok = True
    for idx, s, e, scaling in sectors:
        seg = [r for r in ratios[s:min(e + 1, n)] if r == r]
        if not seg:
            continue
        seg.sort()
        med = seg[len(seg) // 2]
        expect = (scaling * phase) if use_sec else dflt
        good = abs(med - expect) < 0.02
        ok &= good
        print(f'  Sector{idx:<2}{f"{s}-{e}":>14}{scaling:>14.2f}{expect:>8.2f}{med:>12.3f}'
              f'{"  OK" if good else "  <-- 불일치"}')

    print()
    # phase_multiplier != 1.0 은 "공식은 맞지만 yaml 대로 달리고 있지 않다"는 뜻이다.
    # 첫 판에서 이 경우에 OK 만 찍는 바람에, 0.84 로 달리는 차를 정상으로 오독했다.
    if ok and use_sec and phase is not None and abs(phase - 1.0) > 1e-6:
        print(f'  => 공식은 맞지만 phase_multiplier 가 {phase} 라서 yaml 값대로 달리지 않습니다.')
        print(f'     yaml 의 scaling 이 전부 {phase} 배로 줄어듭니다 (1.2 -> {1.2*phase:.2f}).')
        print('     quali 라면 launch 의 speed_scale 누수를 확인하세요 (base_system 과 이름이 같음).')
        print('     즉시 교정: ros2 param set /speed_sector_tuner phase_multiplier 1.0')
    elif ok:
        print('  => 맵의 speed_scaling.yaml 이 그대로 반영되고 있습니다.')
    elif use_sec and all(abs(r - (dflt or 0)) < 0.02 for r in ratios if r == r):
        print('  => use_sector_scaling 은 True 인데 실측이 default_scaling 과 같습니다.')
        print('     tuner 가 재스케일하지 않았다는 뜻입니다 (버그).')
    else:
        print('  => 기대값과 다릅니다. 위 표의 불일치 행을 보고하세요.')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
