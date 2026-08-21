#!/usr/bin/env python3
"""Check cross-package static-avoidance configuration invariants."""

import argparse
import json
import sys
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory


def read_yaml(path):
    with path.open(encoding='utf-8') as stream:
        return yaml.safe_load(stream) or {}


class Results:
    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def check(self, label, left, right, *, strict=False):
        passed = left > right if strict else left >= right
        status = 'OK' if passed else 'FAIL'
        self.failures += int(not passed)
        operator = '>' if strict else '>='
        print(f'[{status:4}] {label}: {left:.3f} {operator} {right:.3f}')

    def warn(self, label, left, right):
        passed = left <= right
        status = 'OK' if passed else 'WARN'
        self.warnings += int(not passed)
        print(f'[{status:4}] {label}: {left:.3f} <= {right:.3f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', help='also evaluate the shared track_length/3 cap')
    args = parser.parse_args()

    stack = Path(get_package_share_directory('stack_master'))
    state = Path(get_package_share_directory('state_machine'))
    planner = read_yaml(stack / 'config/static_avoidance_planner.yaml') \
        ['static_avoidance_planner']['ros__parameters']
    state_params = read_yaml(stack / 'config/state_machine_params.yaml') \
        ['state_machine']['ros__parameters']
    perception = read_yaml(stack / 'config/opponent_tracker_params.yaml')
    static = read_yaml(state / 'config/planners/static_avoidance_planner.yaml')
    tracking = read_yaml(state / 'config/planners/global_tracking.yaml')

    result = Results()
    print('Static-avoidance configuration invariants')
    result.check('planner free gap vs state-machine free gap',
                 float(planner['min_free_dist_m']), float(static['lateral_width_m']))
    blocked_width = float(state_params['gb_ego_width_m']) / 2.0 \
        + float(tracking['lateral_width_m'])
    result.check('planner raceline clearance vs blocked-line threshold',
                 float(planner['raceline_clearance_m']), blocked_width)
    result.check('published path length vs state-machine horizon',
                 float(planner['min_path_end_m']), float(static['max_horizon']))
    result.check('preferred evasion vs hard geometric floor',
                 float(planner['evasion_distance']),
                 float(planner['ego_width_m']) / 2.0
                 + max(float(planner['min_free_dist_m']), 0.025), strict=True)

    same_width = abs(float(planner['ego_width_m'])
                     - float(state_params['gb_ego_width_m'])) < 1e-9
    print(f'[{"OK" if same_width else "FAIL":4}] planner/state-machine ego width match')
    result.failures += int(not same_width)

    lookahead = float(planner['lookahead'])
    result.warn('lookahead vs tracking front range', lookahead,
                float(perception['tracking']['ros__parameters']['dist_infront']))
    result.warn('lookahead vs detector range', lookahead,
                float(perception['detect']['ros__parameters']['max_viewing_distance']))

    if args.map:
        waypoint_file = stack / 'maps' / args.map / 'global_waypoints.json'
        with waypoint_file.open(encoding='utf-8') as stream:
            waypoints = json.load(stream)['global_traj_wpnts_iqp']['wpnts']
        track_length = max(float(point['s_m']) for point in waypoints)
        cap = track_length / 3.0
        # Both planner and kyumin state machine apply this same runtime cap.
        result.check('effective path end vs effective state-machine horizon',
                     min(float(planner['min_path_end_m']), cap),
                     min(float(static['max_horizon']), cap))

    print(f'Result: {result.failures} failure(s), {result.warnings} warning(s)')
    return 1 if result.failures else 0


if __name__ == '__main__':
    sys.exit(main())
