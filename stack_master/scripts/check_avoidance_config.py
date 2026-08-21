#!/usr/bin/env python3
"""check_avoidance_config.py — cross-file invariants for static avoidance.

The static planner and the state machine judge the SAME path with the same
arithmetic, from FOUR different files. Break one pairing and the failure is not
an error message, it is a car that behaves oddly on the track:

  * planner clearance < state machine's        -> the planner publishes paths the
    lateral_width_m                               state machine refuses; the car
                                                  trails the obstacle forever.
  * raceline_clearance_m too small             -> the state machine asks for an
                                                  avoidance the planner thinks is
                                                  unnecessary; same stall.
  * min_path_end_m < max_horizon               -> any obstacle past the path's end
                                                  reads as blocking.
  * evasion_distance below the hard floor      -> the relaxation ladder's clearance
                                                  rungs all collapse to the same
                                                  number and stop relaxing.

No ROS needed: this reads the yamls straight off disk.

  ros2 run stack_master check_avoidance_config.py
  ros2 run stack_master check_avoidance_config.py --map 0820test2
"""

import argparse
import json
import os
import sys

import yaml

# Mirrors static_avoidance_node.py. Keep in step if those change.
SAFETY_MARGIN_M = 0.025
TRACK_FRACTION = 1.0 / 3.0

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))


def load(rel):
    p = os.path.join(REPO, rel)
    if not os.path.exists(p):
        print(f'  ! 파일 없음: {rel}')
        return {}, p
    with open(p) as f:
        return (yaml.safe_load(f) or {}), p


class Check:
    def __init__(self):
        self.fail = 0
        self.warn = 0

    def ok(self, name, lhs_s, lhs, op, rhs_s, rhs, note=''):
        good = (lhs >= rhs) if op == '>=' else (lhs > rhs)
        margin = lhs - rhs
        tag = 'OK  ' if good else 'FAIL'
        if not good:
            self.fail += 1
        elif 0 <= margin < 0.01:
            tag = 'TIGHT'
            self.warn += 1
        print(f'  [{tag:5}] {name}')
        print(f'          {lhs_s} = {lhs:.3f}  {op}  {rhs_s} = {rhs:.3f}   (여유 {margin:+.3f} m)')
        if note:
            print(f'          {note}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default=None, help='트랙 길이 의존 검사를 수행할 맵 이름')
    args = ap.parse_args()

    pl, pl_p = load('stack_master/config/static_avoidance_planner.yaml')
    pl = (pl.get('static_avoidance_planner') or {}).get('ros__parameters') or {}
    sm_static, _ = load('state_machine/config/planners/static_avoidance_planner.yaml')
    sm_track, _ = load('state_machine/config/planners/global_tracking.yaml')
    smp, _ = load('stack_master/config/state_machine_params.yaml')
    smp = (smp.get('state_machine') or {}).get('ros__parameters') or {}
    perc, _ = load('stack_master/config/opponent_tracker_params.yaml')
    det = (perc.get('detect') or {}).get('ros__parameters') or {}
    trk = (perc.get('tracking') or {}).get('ros__parameters') or {}

    c = Check()
    print('\n=== static avoidance 교차 파일 불변식 ===\n')

    ego_w = float(pl.get('ego_width_m', 0))
    mfd = float(pl.get('min_free_dist_m', 0))
    sm_lat = float(sm_static.get('lateral_width_m', 0))
    gt_lat = float(sm_track.get('lateral_width_m', 0))
    gb_w = float(smp.get('gb_ego_width_m', 0))

    c.ok('1. 플래너 자유폭 >= 상태기계 자유폭',
         'min_free_dist_m', mfd, '>=', 'state_machine lateral_width_m', sm_lat,
         '깨지면: 플래너가 만든 경로를 상태기계가 거부 -> TRAILING 고착')

    c.ok('2. 레이스라인 클리어런스 >= 상태기계의 "막힘" 기준',
         'raceline_clearance_m', float(pl.get('raceline_clearance_m', 0)), '>=',
         'gb_ego_width_m/2 + global_tracking.lateral_width_m', gb_w / 2 + gt_lat,
         '깨지면: 상태기계는 회피를 요구하는데 플래너는 불필요하다고 판단 -> 영원히 추종')

    c.ok('3. 경로 끝 >= 상태기계 판단 지평',
         'min_path_end_m', float(pl.get('min_path_end_m', 0)), '>=',
         'state_machine max_horizon', float(sm_static.get('max_horizon', 0)),
         '깨지면: 경로 끝 너머의 장애물이 전부 "막힘"으로 읽힘')

    floor = ego_w / 2 + max(mfd, SAFETY_MARGIN_M)
    ev = float(pl.get('evasion_distance', 0))
    if ev <= floor:
        c.fail += 1
    print(f'  [{"OK  " if ev > floor else "FAIL "}] 4. 완화 사다리가 실제로 완화되는가')
    print(f'          evasion_distance = {ev:.3f}   하한(min_evasion_m) = {floor:.3f}')
    if ev <= floor:
        print(f'          ! evasion_distance 가 하한 이하다. 사다리의 클리어런스 단'
              f'(x1.0/x0.7/x0.0)이')
        print(f'            전부 {floor:.3f} 으로 뭉개져 클리어런스 완화가 무효가 된다.')
        print(f'            선호 클리어런스를 쓰려면 {floor:.3f} 보다 크게 둘 것.')
    else:
        print(f'          사다리: x1.0 -> {max(ev,floor):.3f}, x0.7 -> {max(0.7*ev,floor):.3f}, '
              f'x0.0 -> {floor:.3f}')

    print(f'\n  [{"OK  " if abs(ego_w - gb_w) < 1e-9 else "FAIL"}] 5. 차폭 일치')
    print(f'          planner ego_width_m = {ego_w:.3f}   state_machine gb_ego_width_m = {gb_w:.3f}')
    if abs(ego_w - gb_w) > 1e-9:
        c.fail += 1

    print('\n=== 통과에 필요한 실제 폭 ===\n')
    bm = float(pl.get('boundary_margin', 0))
    need = floor + bm
    print(f'  장애물 가장자리 -> 경로 중심선 : {floor:.3f} m  (차 반폭 {ego_w/2:.3f} + 자유폭 {max(mfd,SAFETY_MARGIN_M):.3f})')
    print(f'  경로 중심선 -> 트랙 경계       : {bm:.3f} m  (boundary_margin)')
    print(f'  ------------------------------------------')
    print(f'  필요한 최소 여유               : {need:.3f} m')
    print(f'  차가 벽을 지나는 실제 간격     : {bm - ego_w/2:.3f} m')

    print('\n=== 인지 계층과의 정합 ===\n')
    la = float(pl.get('lookahead', 0))
    for nm, v in (('tracking.dist_infront', trk.get('dist_infront')),
                  ('detect.max_viewing_distance', det.get('max_viewing_distance'))):
        if v is None:
            continue
        s = 'OK  ' if la <= float(v) else 'WARN'
        if la > float(v):
            c.warn += 1
        print(f'  [{s}] lookahead {la:.1f} m  <=  {nm} {float(v):.1f} m')
    print(f'  detect rate {det.get("rate_detect")} Hz / tracking rate {trk.get("rate_tracking")} Hz '
          f'/ planner {pl.get("rate_hz")} Hz')

    if args.map:
        print(f'\n=== 짧은 트랙 캡 ({args.map}) ===\n')
        jp = os.path.join(REPO, 'stack_master/maps', args.map, 'global_waypoints.json')
        if os.path.exists(jp):
            with open(jp) as f:
                w = json.load(f)['global_traj_wpnts_iqp']['wpnts']
            L = max(x['s_m'] for x in w)
            cap = L * TRACK_FRACTION
            eff = min(float(pl.get('min_path_end_m', 0)), cap)
            mh = float(sm_static.get('max_horizon', 0))
            print(f'  트랙 길이 {L:.2f} m -> track/3 = {cap:.2f} m')
            print(f'  min_path_end_m {pl.get("min_path_end_m")} 는 런타임에 {eff:.2f} m 로 캡된다')
            print(f'  하지만 state_machine max_horizon {mh:.1f} 은 캡되지 않는다')
            if eff < mh:
                c.fail += 1
                print(f'  [FAIL ] 불변식 3 이 이 맵에서 깨진다: {eff:.2f} < {mh:.1f}')
                print(f'          "정적 장애물이 경로 끝 너머" 로 읽혀 TRAILING 에 고착된다.')
                print(f'          이 맵에서 달리기 전에 max_horizon 을 {eff:.2f} 이하로 낮출 것.')
            else:
                print(f'  [OK   ] {eff:.2f} >= {mh:.1f}')
        else:
            print(f'  ! 맵을 찾을 수 없음: {jp}')

    print(f'\n=== 결과: 실패 {c.fail}건, 경고 {c.warn}건 ===\n')
    sys.exit(1 if c.fail else 0)


if __name__ == '__main__':
    main()
