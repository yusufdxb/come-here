#!/usr/bin/env python3
"""Come Here ANY: investigate the GO2's native obstacle avoidance, one explicit step at a time.

Run on the Jetson with the ROS environment sourced and NOTHING else commanding
the robot (no come-here / FETCH / Nav2 launch). Every run appends one JSON
record to ~/come_here_trials/native_avoid_probe.jsonl.

Read-only (sends only query RPCs; never motion):
  status                 CheckMode, Sport + obstacles_avoid server api version (internal
                         RPC api 1), obstacles_avoid SwitchGet, /multiplestate
                         obstaclesAvoidSwitch, /sportmodestate mode/gait/height/velocity/
                         range_obstacle, response-topic publisher counts
  watch --seconds N      record /multiplestate, SwitchGet (1 Hz), /sportmodestate and
                         /wirelesscontroller keys while the operator presses remote
                         buttons (which button toggles which state?)

State changes (need --supervised: operator present, robot standing, remote in hand;
each watches body velocity for --watch-s and sends StopMove if it exceeds 0.1 m/s):
  freeavoid on|off       Sport 2048 FreeAvoid({"data": bool})
  oa-switch on|off       obstacles_avoid SwitchSet 1001 + SwitchGet 1002 read-back
  api-control take|release   obstacles_avoid UseRemoteCommandFromApi 1004
  remote-check           take API control, ask the operator to push a stick, record whether
                         /wirelesscontroller still reports it (the come-here manual
                         override depends on it), release

Always allowed:
  stop                   obstacles_avoid Move(0,0,0), Sport StopMove, release API control

Motion (needs --i-understand-this-moves-the-robot, open floor, operator with remote):
  move --backend {freeavoid,oa} --vx V [--vy V] [--yaw W] --seconds S
                         enable the backend, stream the command at 20 Hz for S seconds
                         (S <= 2.0, |vx| <= 0.6, |vy| <= 0.3, |yaw| <= 1.0), then stop and
                         restore; records measured velocity, odometry displacement
                         (forward / lateral / yaw), min speed while commanded (braking?)
                         and range_obstacle. Ctrl+C stops at once.

NOTHING here decides what FreeAvoid "means": it records what the robot did.
"""

import argparse
import datetime
import json
import math
import os
import signal
import sys
import time

TRIAL_DIR = os.path.expanduser('~/come_here_trials')
LOG_NAME = 'native_avoid_probe.jsonl'

MAX_SECONDS = 2.0
MAX_VX = 0.6
MAX_VY = 0.3
MAX_YAW = 1.0
RUNAWAY_SPEED = 0.1        # m/s: after a state change, faster than this -> StopMove
MOTION_FLAG = '--i-understand-this-moves-the-robot'


# -- pure helpers (unit-tested in come_here_bringup/test/test_native_avoid_probe.py) --

def check_motion_args(vx, vy, yaw, seconds):
    """Return a list of refusal reasons (empty = allowed)."""
    problems = []
    for name, value, cap in (('vx', vx, MAX_VX), ('vy', vy, MAX_VY), ('yaw', yaw, MAX_YAW),
                             ('seconds', seconds, MAX_SECONDS)):
        if not math.isfinite(value):
            problems.append(f'{name} is not finite')
        elif abs(value) > cap:
            problems.append(f'|{name}|={abs(value)} exceeds {cap}')
    if seconds <= 0.0:
        problems.append('seconds must be > 0')
    if vx < 0.0:
        problems.append('backward motion is not part of this probe')
    return problems


def body_frame_displacement(start, end):
    """(forward_m, lateral_m, yaw_change_rad) of pose ``end`` in the frame of ``start``.

    Poses are (x, y, yaw) in the odometry frame; lateral is positive to the left.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    c, s = math.cos(start[2]), math.sin(start[2])
    dyaw = math.atan2(math.sin(end[2] - start[2]), math.cos(end[2] - start[2]))
    return dx * c + dy * s, -dx * s + dy * c, dyaw


def speed_stats(samples, t0, t1):
    """max / min planar speed of SportModeState samples [(t, vx, vy)] inside [t0, t1]."""
    speeds = [math.hypot(vx, vy) for t, vx, vy in samples if t0 <= t <= t1]
    if not speeds:
        return None
    return {'max_speed': round(max(speeds), 3), 'min_speed': round(min(speeds), 3),
            'mean_speed': round(sum(speeds) / len(speeds), 3), 'samples': len(speeds)}


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def append_record(record, directory=TRIAL_DIR):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, LOG_NAME)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + '\n')
        f.flush()
        os.fsync(f.fileno())
    return path


# -- ROS side ----------------------------------------------------------------

class Probe:
    def __init__(self):
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        from std_msgs.msg import String
        from unitree_api.msg import Request, Response
        from unitree_go.msg import SportModeState, WirelessController

        self.rclpy = rclpy
        self.Request = Request
        rclpy.init()
        n = self.node = rclpy.create_node('native_avoid_probe')
        self.pubs = {
            'sport': n.create_publisher(Request, '/api/sport/request', 10),
            'obstacles_avoid': n.create_publisher(Request, '/api/obstacles_avoid/request', 10),
            'motion_switcher': n.create_publisher(Request, '/api/motion_switcher/request', 10),
        }
        self.responses = {}
        for service in self.pubs:
            n.create_subscription(Response, f'/api/{service}/response',
                                  lambda m, s=service: self._on_response(s, m), 50)
        self.state = []       # (t, vx, vy, yaw_speed, mode, gait, body_height, range_obstacle)
        self.pose = None      # (t, x, y, yaw)
        self.multistate = []  # (t, obstaclesAvoidSwitch)
        self.remote = []      # (t, lx, ly, rx, ry, keys)
        n.create_subscription(SportModeState, '/sportmodestate', self._on_state,
                              qos_profile_sensor_data)
        n.create_subscription(Odometry, '/utlidar/robot_odom', self._on_odom,
                              qos_profile_sensor_data)
        n.create_subscription(String, '/multiplestate', self._on_multi, 10)
        n.create_subscription(WirelessController, '/wirelesscontroller', self._on_remote,
                              qos_profile_sensor_data)
        self._next_id = int(time.time() * 1000) % 1_000_000_000 + 1_000_000_000
        self.sent = []

    # callbacks
    def _on_response(self, service, m):
        self.responses[(service, int(m.header.identity.id))] = (
            int(m.header.status.code), m.data)

    def _on_state(self, m):
        self.state.append((time.monotonic(), float(m.velocity[0]), float(m.velocity[1]),
                           float(m.yaw_speed), int(m.mode), int(m.gait_type),
                           float(m.body_height), [round(float(r), 3) for r in m.range_obstacle]))
        if len(self.state) > 20000:
            del self.state[:10000]

    def _on_odom(self, m):
        p = m.pose.pose
        self.pose = (time.monotonic(), p.position.x, p.position.y,
                     yaw_from_quaternion(p.orientation))

    def _on_multi(self, m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        if isinstance(d, dict) and 'obstaclesAvoidSwitch' in d:
            self.multistate.append((time.monotonic(), bool(d['obstaclesAvoidSwitch'])))

    def _on_remote(self, m):
        self.remote.append((time.monotonic(), round(float(m.lx), 2), round(float(m.ly), 2),
                            round(float(m.rx), 2), round(float(m.ry), 2), int(m.keys)))

    # plumbing
    def spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.01)

    def send(self, service, api_id, params=None, noreply=False, label=''):
        msg = self.Request()
        self._next_id += 1
        msg.header.identity.id = self._next_id
        msg.header.identity.api_id = api_id
        msg.header.lease.id = 0
        msg.header.policy.priority = 0
        msg.header.policy.noreply = noreply
        msg.parameter = '' if params is None else json.dumps(params)
        msg.binary = []
        self.pubs[service].publish(msg)
        self.sent.append({'t': round(time.monotonic(), 3), 'service': service, 'api_id': api_id,
                          'params': params, 'label': label})
        return self._next_id

    def call(self, service, api_id, params=None, timeout=2.0, label=''):
        """Send and wait for the reply: (code, data) or (None, None) on timeout."""
        rid = self.send(service, api_id, params, False, label)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.02)
            if (service, rid) in self.responses:
                return self.responses[(service, rid)]
        return None, None

    def snapshot(self):
        s = self.state[-1] if self.state else None
        return {
            'sportmodestate': None if s is None else {
                'vx': round(s[1], 3), 'vy': round(s[2], 3), 'yaw_speed': round(s[3], 3),
                'mode': s[4], 'gait_type': s[5], 'body_height': round(s[6], 3),
                'range_obstacle': s[7]},
            'obstaclesAvoidSwitch': self.multistate[-1][1] if self.multistate else None,
            'odom': None if self.pose is None else [round(v, 3) for v in self.pose[1:]],
        }

    def stop(self):
        self.send('obstacles_avoid', 1003, {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'mode': 0}, True,
                  'oa_move_zero')
        self.send('sport', 1003, None, False, 'stop_move')
        self.spin(0.05)
        self.send('obstacles_avoid', 1004, {'is_remote_commands_from_api': False}, False,
                  'release_api_control')
        self.spin(0.2)

    def watch_velocity(self, seconds):
        """Watch for motion after a state change; StopMove if it runs away."""
        t0 = time.monotonic()
        stopped = False
        end = t0 + seconds
        while time.monotonic() < end:
            self.spin(0.05)
            recent = [math.hypot(vx, vy) for t, vx, vy, *_ in self.state if t >= t0]
            if recent and max(recent) > RUNAWAY_SPEED and not stopped:
                self.send('sport', 1003, None, False, 'backstop_stop_move')
                stopped = True
        stats = speed_stats([(t, vx, vy) for t, vx, vy, *_ in self.state], t0, end)
        return {'window_s': seconds, 'speed': stats, 'backstop_stop_sent': stopped}

    def close(self):
        self.node.destroy_node()
        self.rclpy.try_shutdown()


def cmd_status(p, args, rec):
    p.spin(1.0)
    code, data = p.call('motion_switcher', 1001, None, label='check_mode')
    rec['check_mode'] = {'code': code, 'data': data}
    if not args.no_version:
        for service in ('sport', 'obstacles_avoid'):
            code, data = p.call(service, 1, {}, label=f'{service}_api_version')
            rec[f'{service}_server_version'] = {'code': code, 'data': data}
    code, data = p.call('obstacles_avoid', 1002, {}, label='switch_get')
    rec['oa_switch_get'] = {'code': code, 'data': data}
    rec['response_publishers'] = {
        s: p.node.count_publishers(f'/api/{s}/response') for s in p.pubs}
    rec['snapshot'] = p.snapshot()


def cmd_watch(p, args, rec):
    events = []
    end = time.monotonic() + args.seconds
    last_get = 0.0
    print(f'watching for {args.seconds:.0f} s: press the remote button(s) now, one at a time')
    while time.monotonic() < end:
        p.spin(0.05)
        if time.monotonic() - last_get >= 1.0:
            last_get = time.monotonic()
            code, data = p.call('obstacles_avoid', 1002, {}, timeout=0.8, label='switch_get')
            events.append({'t': round(last_get, 2), 'switch_get': data, 'code': code,
                           'multistate': p.multistate[-1][1] if p.multistate else None})
    rec['switch_timeline'] = events
    rec['multistate_changes'] = _changes(p.multistate)
    rec['remote_keys'] = [r for r in p.remote if r[5] != 0][:400]
    rec['mode_gait_changes'] = _changes([(t, (m, g)) for t, _a, _b, _c, m, g, _h, _r in p.state])


def _changes(series):
    out, last = [], object()
    for t, v in series:
        if v != last:
            out.append((round(t, 2), v))
            last = v
    return out


def cmd_state_change(p, args, rec):
    what = args.command
    if what == 'freeavoid':
        code, data = p.call('sport', 2048, {'data': args.value == 'on'}, label='free_avoid')
        rec['free_avoid'] = {'value': args.value, 'code': code, 'data': data}
    elif what == 'oa-switch':
        code, data = p.call('obstacles_avoid', 1001, {'enable': args.value == 'on'},
                            label='switch_set')
        rec['switch_set'] = {'value': args.value, 'code': code, 'data': data}
        rec['switch_get_after'] = p.call('obstacles_avoid', 1002, {}, label='switch_get')
    elif what == 'api-control':
        code, data = p.call('obstacles_avoid', 1004,
                            {'is_remote_commands_from_api': args.value == 'take'},
                            label='use_remote_command_from_api')
        rec['api_control'] = {'value': args.value, 'code': code, 'data': data}
    rec['after'] = p.watch_velocity(args.watch_s)
    rec['snapshot'] = p.snapshot()


def cmd_remote_check(p, args, rec):
    code, data = p.call('obstacles_avoid', 1004, {'is_remote_commands_from_api': True},
                        label='take_api_control')
    rec['take'] = {'code': code, 'data': data}
    n0 = len(p.remote)
    print(f'API control requested (code {code}). Push the LEFT stick fully forward and back '
          f'within {args.seconds:.0f} s.')
    watch = p.watch_velocity(args.seconds)
    moved = [r for r in p.remote[n0:] if max(abs(r[1]), abs(r[2]), abs(r[3]), abs(r[4])) > 0.2]
    rec['remote_messages'] = len(p.remote) - n0
    rec['remote_stick_events'] = len(moved)
    rec['robot_motion'] = watch
    code, data = p.call('obstacles_avoid', 1004, {'is_remote_commands_from_api': False},
                        label='release_api_control')
    rec['release'] = {'code': code, 'data': data}
    rec['verdict'] = ('override_signal_present' if moved
                      else 'NO_STICK_EVENTS_WHILE_API_CONTROLLED (hardware NO-GO for '
                           'native_use_remote_command_from_api until explained)')


def cmd_move(p, args, rec):
    problems = check_motion_args(args.vx, args.vy, args.yaw, args.seconds)
    if problems:
        raise SystemExit('refused: ' + '; '.join(problems))
    p.spin(1.0)
    code, data = p.call('motion_switcher', 1001, None, label='check_mode')
    if code != 0 or '"mcf"' not in str(data).replace(' ', ''):
        raise SystemExit(f'refused: motion mode is not verified mcf ({code}, {data})')
    snap = p.snapshot()
    height = (snap['sportmodestate'] or {}).get('body_height') or 0.0
    if height < 0.28:
        raise SystemExit(f'refused: robot not standing (body_height {height})')
    if p.pose is None:
        raise SystemExit('refused: no /utlidar/robot_odom')
    rec['before'] = snap
    backend = args.backend
    if backend == 'freeavoid':
        rec['enable'] = p.call('sport', 2048, {'data': True}, label='free_avoid_on')
    else:
        rec['switch_initial'] = p.call('obstacles_avoid', 1002, {}, label='switch_get')
        rec['enable'] = p.call('obstacles_avoid', 1001, {'enable': True}, label='switch_on')
        rec['switch_verify'] = p.call('obstacles_avoid', 1002, {}, label='switch_get')
        rec['api_control'] = p.call('obstacles_avoid', 1004,
                                    {'is_remote_commands_from_api': True}, label='take')
    if rec['enable'][0] != 0:
        raise SystemExit(f'refused: enable failed {rec["enable"]}')
    p.spin(0.5)
    start_pose = p.pose[1:]
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < args.seconds:
            if backend == 'freeavoid':
                p.send('sport', 1008, {'x': args.vx, 'y': args.vy, 'z': args.yaw}, False, 'move')
            else:
                p.send('obstacles_avoid', 1003,
                       {'x': args.vx, 'y': args.vy, 'yaw': args.yaw, 'mode': 0}, True, 'oa_move')
            p.spin(0.05)
    finally:
        t1 = time.monotonic()
        p.stop()
    p.spin(1.0)
    end_pose = p.pose[1:]
    fwd, lat, dyaw = body_frame_displacement(start_pose, end_pose)
    rec['command'] = {'backend': backend, 'vx': args.vx, 'vy': args.vy, 'yaw': args.yaw,
                      'seconds': args.seconds}
    rec['displacement'] = {'forward_m': round(fwd, 3), 'lateral_m': round(lat, 3),
                           'yaw_change_rad': round(dyaw, 3)}
    samples = [(t, vx, vy) for t, vx, vy, *_ in p.state]
    rec['speed_while_commanded'] = speed_stats(samples, t0 + 0.3, t1)
    rec['speed_after_stop'] = speed_stats(samples, t1 + 0.5, t1 + 1.0)
    ranges = [row[7] for row in p.state if row[0] >= t0 and len(row[7]) == 4]
    rec['range_obstacle_min'] = ([round(min(r[i] for r in ranges), 3) for i in range(4)]
                                 if ranges else None)
    if backend == 'freeavoid' and args.restore:
        rec['disable'] = p.call('sport', 2048, {'data': False}, label='free_avoid_off')
    if backend == 'oa' and args.restore and rec['switch_initial'][0] == 0:
        try:
            initial = json.loads(rec['switch_initial'][1])['enable']
            rec['switch_restore'] = p.call('obstacles_avoid', 1001, {'enable': initial},
                                           label='switch_restore')
        except (ValueError, KeyError, TypeError):
            rec['switch_restore'] = 'initial state unreadable: left on'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--note', default='', help='free text stored with the record (setup, obstacle)')
    sub = ap.add_subparsers(dest='command', required=True)
    s = sub.add_parser('status')
    s.add_argument('--no-version', action='store_true', help='skip the api-version queries')
    w = sub.add_parser('watch')
    w.add_argument('--seconds', type=float, default=60.0)
    for name, choices in (('freeavoid', ('on', 'off')), ('oa-switch', ('on', 'off')),
                          ('api-control', ('take', 'release'))):
        c = sub.add_parser(name)
        c.add_argument('value', choices=choices)
        c.add_argument('--supervised', action='store_true', required=True)
        c.add_argument('--watch-s', type=float, default=3.0)
    r = sub.add_parser('remote-check')
    r.add_argument('--supervised', action='store_true', required=True)
    r.add_argument('--seconds', type=float, default=10.0)
    sub.add_parser('stop')
    m = sub.add_parser('move')
    m.add_argument('--backend', choices=('freeavoid', 'oa'), required=True)
    m.add_argument('--vx', type=float, required=True)
    m.add_argument('--vy', type=float, default=0.0)
    m.add_argument('--yaw', type=float, default=0.0)
    m.add_argument('--seconds', type=float, required=True)
    m.add_argument('--no-restore', dest='restore', action='store_false',
                   help='leave FreeAvoid / the switch on afterwards')
    m.add_argument(MOTION_FLAG, dest='motion_ok', action='store_true')
    args = ap.parse_args(argv)
    if args.command == 'move':
        if not args.motion_ok:
            raise SystemExit(f'refused: motion needs {MOTION_FLAG}')
        problems = check_motion_args(args.vx, args.vy, args.yaw, args.seconds)
        if problems:
            raise SystemExit('refused: ' + '; '.join(problems))

    rec = {'command': args.command, 'argv': sys.argv[1:], 'note': args.note,
           'started_at': datetime.datetime.now().isoformat(timespec='seconds')}
    p = Probe()

    def on_signal(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)
    try:
        handler = {'status': cmd_status, 'watch': cmd_watch, 'freeavoid': cmd_state_change,
                   'oa-switch': cmd_state_change, 'api-control': cmd_state_change,
                   'remote-check': cmd_remote_check, 'move': cmd_move,
                   'stop': lambda p_, a, r: p_.stop()}[args.command]
        handler(p, args, rec)
        rec['result'] = 'ok'
    except KeyboardInterrupt:
        rec['result'] = 'interrupted'
        p.stop()
    except SystemExit as exc:
        rec['result'] = f'refused: {exc}'
        if args.command == 'move':
            p.stop()
        print(exc, file=sys.stderr)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        rec['sent'] = p.sent
        rec['ended_at'] = datetime.datetime.now().isoformat(timespec='seconds')
        path = append_record(rec)
        p.close()
    print(json.dumps({k: v for k, v in rec.items() if k != 'sent'}, indent=2, default=str))
    print(f'record appended to {path}')
    return 0 if rec['result'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
