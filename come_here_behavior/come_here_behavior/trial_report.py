"""Trial table for the class demo: one line per trial, with the operator's verdict.

    ros2 run come_here_behavior trial_report                            # table + streak
    ros2 run come_here_behavior trial_report --mark PASS --distance 0.85
    ros2 run come_here_behavior trial_report --mark FAIL --note "stopped at 1.3 m"
    ros2 run come_here_behavior trial_report --mark FAIL --no-trial --note "wake missed"

--mark annotates the most recent trial (or --run RUN_ID). The operator's
PASS/FAIL is the verdict that counts; the automatic stop reason is shown next
to it. A missed wake phrase creates no trial record, so log it with --no-trial.
"""

import argparse
import datetime
import json
import os

from come_here_behavior.trial_log import TRIALS_FILENAME, read_jsonl

MARKS_FILENAME = 'trial_marks.jsonl'
TARGET_STREAK = 5


def build_rows(trials: list, marks: list) -> list:
    by_run = {}
    missed = []
    for mark in marks:
        if mark.get('run_id'):
            by_run[mark['run_id']] = mark  # the latest mark wins
        else:
            missed.append(mark)
    rows = []
    for trial in trials:
        mark = by_run.get(trial.get('run_id'), {})
        rows.append({
            'time': trial.get('started_at') or '',
            'commit': (trial.get('git_commit') or '')[:7],
            'wake': (trial.get('wake') or {}).get('source', 'unknown'),
            'acquired': trial.get('acquire_latency_s') is not None,
            'moved': (trial.get('motion_commands') or 0) > 0,
            'stop': trial.get('stop_reason'),
            'bbox': trial.get('final_bbox_h_frac'),
            'verdict': mark.get('verdict'),
            'distance_m': mark.get('distance_m'),
            'note': mark.get('note', ''),
        })
    for mark in missed:
        rows.append({
            'time': mark.get('marked_at') or '', 'commit': '', 'wake': 'missed',
            'acquired': False, 'moved': False, 'stop': None, 'bbox': None,
            'verdict': mark.get('verdict'), 'distance_m': mark.get('distance_m'),
            'note': mark.get('note', ''),
        })
    rows.sort(key=lambda row: row['time'])
    return rows


def pass_streak(rows: list) -> int:
    """Consecutive PASS verdicts ending at the most recent trial."""
    streak = 0
    for row in reversed(rows):
        if row['verdict'] != 'PASS':
            break
        streak += 1
    return streak


def format_row(index: int, row: dict) -> str:
    distance = '-' if row['distance_m'] is None else f'{row["distance_m"]:.2f}m'
    bbox = '-' if row['bbox'] is None else f'{row["bbox"]:.2f}'
    return (
        f'trial_{index:02d}: {row["verdict"] or "UNMARKED":8s} {row["time"]:19s} '
        f'{row["commit"]:7s} wake={row["wake"]} acquired={"yes" if row["acquired"] else "no"} '
        f'moved={"yes" if row["moved"] else "no"} stop={row["stop"]} bbox={bbox} '
        f'dist={distance} {row["note"]}'
    ).rstrip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', default='~/come_here_trials')
    parser.add_argument('--mark', choices=('PASS', 'FAIL'))
    parser.add_argument('--run', default=None, help='run_id to mark (default: latest trial)')
    parser.add_argument('--distance', type=float, default=None,
                        help='measured final distance to the caller, metres')
    parser.add_argument('--note', default='')
    parser.add_argument('--no-trial', action='store_true',
                        help='record an attempt that produced no trial (e.g. wake missed)')
    args = parser.parse_args(argv)

    directory = os.path.expanduser(args.dir)
    trials = read_jsonl(os.path.join(directory, TRIALS_FILENAME))
    marks_path = os.path.join(directory, MARKS_FILENAME)
    marks = read_jsonl(marks_path)

    if args.mark:
        run_id = None
        if not args.no_trial:
            run_id = args.run or (trials[-1].get('run_id') if trials else None)
            if run_id is None:
                print('No trial to mark. Use --no-trial for an attempt with no record.')
                return 1
        mark = {
            'run_id': run_id, 'verdict': args.mark, 'distance_m': args.distance,
            'note': args.note,
            'marked_at': datetime.datetime.now().isoformat(timespec='seconds'),
        }
        os.makedirs(directory, exist_ok=True)
        with open(marks_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(mark) + '\n')
        marks.append(mark)

    rows = build_rows(trials, marks)
    for index, row in enumerate(rows, start=1):
        print(format_row(index, row))
    passes = sum(1 for row in rows if row['verdict'] == 'PASS')
    fails = sum(1 for row in rows if row['verdict'] == 'FAIL')
    unmarked = len(rows) - passes - fails
    streak = pass_streak(rows)
    print(f'\n{passes} PASS, {fails} FAIL, {unmarked} unmarked. '
          f'Consecutive PASS streak: {streak}/{TARGET_STREAK}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
