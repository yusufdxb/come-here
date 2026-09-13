"""Operator tools: e-stop console commands and the trial report (pure Python)."""

import json

from come_here_behavior.estop_console import ENGAGE, QUIT, RELEASE, STATUS, parse_command
from come_here_behavior.trial_report import build_rows, format_row, main, pass_streak


def test_estop_console_engages_on_enter_and_needs_a_word_to_release():
    assert parse_command('') == ENGAGE
    assert parse_command('e') == ENGAGE
    assert parse_command('release') == RELEASE
    assert parse_command('r') is None  # a single stray key never releases
    assert parse_command('s') == STATUS
    assert parse_command('q') == QUIT


def _trial(run_id, started, success=True, stop='arrived_bbox'):
    return {
        'run_id': run_id, 'started_at': started, 'git_commit': 'abcdef123',
        'wake': {'source': 'audio'}, 'acquire_latency_s': 1.2, 'motion_commands': 30,
        'stop_reason': stop, 'final_bbox_h_frac': 0.77, 'success': success,
    }


def test_rows_join_marks_and_count_the_streak():
    trials = [_trial('a', '2026-09-14T10:00:00'), _trial('b', '2026-09-14T10:02:00'),
              _trial('c', '2026-09-14T10:04:00')]
    marks = [
        {'run_id': 'a', 'verdict': 'FAIL', 'note': 'stopped late'},
        {'run_id': 'b', 'verdict': 'PASS', 'distance_m': 0.85},
        {'run_id': 'c', 'verdict': 'PASS', 'distance_m': 0.9},
    ]
    rows = build_rows(trials, marks)
    assert [row['verdict'] for row in rows] == ['FAIL', 'PASS', 'PASS']
    assert pass_streak(rows) == 2
    assert format_row(2, rows[1]).startswith('trial_02: PASS')


def test_missed_wake_is_a_row_and_breaks_the_streak():
    trials = [_trial('a', '2026-09-14T10:00:00')]
    marks = [{'run_id': 'a', 'verdict': 'PASS'},
             {'run_id': None, 'verdict': 'FAIL', 'note': 'wake missed',
              'marked_at': '2026-09-14T10:05:00'}]
    rows = build_rows(trials, marks)
    assert rows[-1]['wake'] == 'missed'
    assert pass_streak(rows) == 0


def test_mark_writes_a_verdict_for_the_latest_trial(tmp_path, capsys):
    (tmp_path / 'trials.jsonl').write_text(json.dumps(_trial('run-1', '2026-09-14T10:00:00')) + '\n')
    assert main(['--dir', str(tmp_path), '--mark', 'PASS', '--distance', '0.82']) == 0
    marks = [json.loads(line) for line in (tmp_path / 'trial_marks.jsonl').read_text().splitlines()]
    assert marks[0]['run_id'] == 'run-1' and marks[0]['verdict'] == 'PASS'
    assert 'streak: 1/5' in capsys.readouterr().out
