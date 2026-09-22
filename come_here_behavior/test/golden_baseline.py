"""Golden-trace scenarios for the deployed baseline (professor_demo behavior parameters).

Each scenario drives ComeHereFsm through its public inputs on a 10 Hz fake clock with a
small kinematic world, and records every Commands object it returns, serialized to
JSON. The golden file `golden_baseline.jsonl` was generated from the unmodified
baseline (tag baseline/pre-skill-api-f7192c9); test_baseline_golden.py asserts that the
current code reproduces it byte for byte with the skill interface disabled.

    python3 test/golden_baseline.py --write   # only ever run on the baseline commit
"""

import dataclasses
import json
import math
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from come_here_behavior.come_here_fsm import ComeHereFsm, FsmConfig, PersonObservation  # noqa: E402

DEMO_YAML = os.path.join(HERE, '..', '..', 'come_here_bringup', 'config', 'professor_demo.yaml')
GOLDEN = os.path.join(HERE, 'golden_baseline.jsonl')
TICK = 0.1


def demo_config() -> FsmConfig:
    params = yaml.safe_load(open(DEMO_YAML))['behavior_node']['ros__parameters']
    names = {f.name for f in dataclasses.fields(FsmConfig)}
    return FsmConfig(**{k: v for k, v in params.items() if k in names})


def _ser(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _ser(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ser(v) for v in value]
    return value


class World:
    def __init__(self, bearing, distance, visible=True):
        self.px, self.py = distance * math.cos(bearing), distance * math.sin(bearing)
        self.rx = self.ry = self.heading = 0.0
        self.visible = visible

    def step(self, vx, yaw_rate, dt):
        self.heading += yaw_rate * dt
        self.rx += vx * dt * math.cos(self.heading)
        self.ry += vx * dt * math.sin(self.heading)

    def bearing(self):
        b = math.atan2(self.py - self.ry, self.px - self.rx) - self.heading
        return math.atan2(math.sin(b), math.cos(b))

    def observe(self):
        if not self.visible:
            return PersonObservation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1)
        d = math.hypot(self.px - self.rx, self.py - self.ry)
        b = self.bearing()
        if abs(b) > 0.6:   # outside the camera's view
            return PersonObservation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1)
        return PersonObservation(b, d + 1.0, 0.9, 1.0, min(1.0, 0.45 + 0.25 / d), 2.0, 0.1)


class Driver:
    def __init__(self, world, rotate_result=True, face=True):
        self.fsm = ComeHereFsm(demo_config())
        self.world = world
        self.k = 0
        self.cmd = (0.0, 0.0)
        self.trace = []
        self.pending_rotate = None
        self.rotate_result = rotate_result
        self.face = face

    @property
    def t(self):
        return round(self.k * TICK, 6)

    def rec(self, name, cmds):
        if cmds.velocity is not None:
            self.cmd = cmds.velocity
        if cmds.rotate_rad is not None:
            self.pending_rotate = (cmds.rotate_rad, self.t + 1.0)
        self.trace.append({'t': self.t, 'in': name, 'out': _ser(dataclasses.asdict(cmds))})
        if cmds.face_request and self.face:
            self.rec('face', self.fsm.on_face_result(True, self.t, 0.5))

    def wake(self):
        self.rec('wake', self.fsm.on_wake('come here', self.t))

    def direction(self, az, conf=0.9):
        self.rec('direction', self.fsm.on_direction(az, conf, self.t))

    def run(self, seconds, person_every=3):
        for _ in range(round(seconds / TICK)):
            self.world.step(self.cmd[0], self.cmd[1], TICK)
            if self.pending_rotate and self.t >= self.pending_rotate[1]:
                target, _ = self.pending_rotate
                self.pending_rotate = None
                if self.rotate_result:
                    self.world.heading += target
                    self.rec('rotate_result',
                             self.fsm.on_rotate_result(target, target, 'reached', self.t))
            if self.k % person_every == 0:
                self.rec('person', self.fsm.on_person(self.world.observe(), self.t))
            self.rec('tick', self.fsm.tick(self.t))
            self.k += 1


def _voice(d, az_offset=0.0):
    d.direction(d.world.bearing() + az_offset)
    d.wake()


def scenarios():
    def ahead():
        d = Driver(World(0.05, 2.4))
        _voice(d)
        d.run(30)
        d.rec('reset', d.fsm.on_reset(d.t))
        d.run(3)
        return d

    def side_turn():
        d = Driver(World(1.2, 2.4))
        _voice(d)
        d.run(35)
        return d

    def no_direction():
        d = Driver(World(0.1, 2.4))
        d.wake()
        d.run(8)
        return d

    def turn_no_result():
        d = Driver(World(1.2, 2.4), rotate_result=False)
        _voice(d)
        d.run(15)
        return d

    def nobody_relisten():
        d = Driver(World(0.1, 2.4, visible=False))
        _voice(d)
        d.run(40)
        return d

    def wrong_bearing_search():
        d = Driver(World(-1.4, 2.4))
        _voice(d, az_offset=+1.4)
        d.run(45)
        return d

    def caller_speaks_again():
        d = Driver(World(-1.3, 2.4))
        _voice(d, az_offset=+1.2)       # a wrong first bearing
        for _ in range(8):
            d.run(5)
            d.direction(d.world.bearing())  # the caller speaks again
        return d

    def lost_and_reacquire():
        d = Driver(World(0.05, 2.6))
        _voice(d)
        d.run(3.0)
        d.world.visible = False
        d.run(0.8)
        d.world.visible = True
        d.run(25)
        return d

    def walk_budget():
        d = Driver(World(0.02, 4.5))
        _voice(d)
        d.run(30)
        return d

    def estop_mid_walk():
        d = Driver(World(0.05, 2.6))
        _voice(d)
        d.run(3.0)
        d.rec('estop', d.fsm.on_estop(True, d.t))
        d.run(3)
        d.rec('estop', d.fsm.on_estop(False, d.t))
        d.run(2)
        return d

    def estop_in_done():
        d = Driver(World(0.05, 2.4))
        _voice(d)
        d.run(30)
        d.rec('estop', d.fsm.on_estop(True, d.t))
        d.run(2)
        return d

    def shutdown_mid_walk():
        d = Driver(World(0.05, 2.6))
        _voice(d)
        d.run(3.0)
        d.rec('shutdown', d.fsm.shutdown(d.t))
        return d

    def wake_ignored_while_active():
        d = Driver(World(0.05, 2.6))
        _voice(d)
        d.run(2.0)
        d.wake()
        d.run(2.0)
        return d

    return {f.__name__: f for f in (ahead, side_turn, no_direction, turn_no_result,
                                    nobody_relisten, wrong_bearing_search, caller_speaks_again,
                                    lost_and_reacquire,
                                    walk_budget, estop_mid_walk, estop_in_done,
                                    shutdown_mid_walk, wake_ignored_while_active)}


def generate() -> list:
    lines = []
    for name, fn in scenarios().items():
        d = fn()
        for row in d.trace:
            lines.append(json.dumps({'scenario': name, **row}, sort_keys=True))
    return lines


if __name__ == '__main__':
    lines = generate()
    if '--write' in sys.argv:
        open(GOLDEN, 'w').write('\n'.join(lines) + '\n')
    names = {}
    for line in lines:
        s = json.loads(line)['scenario']
        names[s] = names.get(s, 0) + 1
    print(len(lines), names)
