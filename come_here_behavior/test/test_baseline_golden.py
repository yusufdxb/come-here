"""Baseline parity: the deployed Come Here behavior is unchanged, byte for byte.

golden_baseline.jsonl was generated from the unmodified baseline commit (tag
baseline/pre-skill-api-f7192c9) with the professor_demo behavior parameters. The skill
interface is off by default, so every Commands object the state machine emits in every
scenario must equal the golden record exactly.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import golden_baseline  # noqa: E402


def test_default_mode_reproduces_the_baseline_golden_trace():
    expected = open(golden_baseline.GOLDEN).read().splitlines()
    actual = golden_baseline.generate()
    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a == e, f'line {i} differs:\n  got      {a[:300]}\n  expected {e[:300]}'


def test_golden_covers_the_deployed_paths():
    rows = [json.loads(line) for line in open(golden_baseline.GOLDEN)]
    reasons = {r['out']['trial_summary']['stop_reason'] for r in rows
               if r['out']['trial_summary']}
    assert {'no_direction', 'turn_no_result', 'estop', 'shutdown'} <= reasons
    assert any(r['out']['sit'] for r in rows), 'the sit path is exercised'
    assert any(r['out']['rotate_rad'] is not None for r in rows), 'a voice turn is exercised'
