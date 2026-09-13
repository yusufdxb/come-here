"""Trial evidence log: one JSON object per line, appended and fsynced.

No database. Each record is self-describing (git revision, config snapshot,
wake detail, trial summary) so a results table can be rebuilt from the file
alone. See ``trial_report`` for summarizing and annotating trials.
"""

import json
import os
import subprocess

TRIALS_FILENAME = 'trials.jsonl'


def git_revision(path: str) -> dict:
    """Return ``{'commit': sha or 'unknown', 'dirty': bool or None}`` for the
    git checkout containing ``path``. Never raises."""
    directory = path if os.path.isdir(path) else os.path.dirname(os.path.realpath(path))
    try:
        commit = subprocess.run(
            ['git', '-C', directory, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=2.0, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ['git', '-C', directory, 'status', '--porcelain', '--untracked-files=no'],
            capture_output=True, text=True, timeout=2.0, check=True,
        ).stdout
        return {'commit': commit, 'dirty': bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {'commit': 'unknown', 'dirty': None}


class TrialLogWriter:
    def __init__(self, directory: str):
        self.directory = os.path.expanduser(directory)
        os.makedirs(self.directory, exist_ok=True)
        self.path = os.path.join(self.directory, TRIALS_FILENAME)

    def append(self, record: dict) -> str:
        line = json.dumps(record, sort_keys=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
            f.flush()
            # The payload Jetson can lose power mid-session; make the record durable.
            os.fsync(f.fileno())
        return self.path


def read_jsonl(path: str) -> list:
    """Read a JSONL file, skipping blank or corrupt lines."""
    records = []
    if not os.path.isfile(path):
        return records
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
