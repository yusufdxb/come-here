"""Motion-authority grant gate and e-stop reassert policy (pure Python, no ROS).

A guardian publishes a JSON grant naming which controller may command the
robot: {"v":1,"owner":"come_here"|"nav2"|null,"epoch":int,"guardian":str,
"state":str,"mode":str}. The bridge forwards non-stop Sport requests only while
a fresh grant names it. With ``enabled=False`` (no authority topic) the gate is
always owned and never emits transitions, so legacy behaviour is unchanged.
"""

import json
import math
import threading

REVOKED = 'REVOKED'
ACQUIRED = 'ACQUIRED'


def parse_grant(raw):
    """Return the grant dict, or None if malformed, wrong version or wrong types."""
    try:
        data = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else None
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    v = data.get('v')
    if isinstance(v, bool) or v != 1 or not isinstance(v, int):
        return None
    owner = data.get('owner')
    if owner is not None and not isinstance(owner, str):
        return None
    epoch = data.get('epoch')
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        return None
    for key in ('guardian', 'state', 'mode'):
        if not isinstance(data.get(key), str):
            return None
    return data


class AuthorityGate:
    def __init__(self, name, timeout_s, enabled=True):
        if not (math.isfinite(timeout_s) and timeout_s > 0.0):
            raise ValueError(f'grant_timeout_s must be > 0, got {timeout_s}')
        self.name = name
        self.timeout_s = float(timeout_s)
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._grant = None
        self._grant_t = None
        self._was_owned = False
        self._token = None

    def on_grant(self, raw, now):
        grant = parse_grant(raw)
        if grant is None:
            return False
        with self._lock:
            g = self._grant
            if g is not None and grant['guardian'] == g['guardian'] \
                    and grant['epoch'] < g['epoch']:
                return False  # late grant from an older epoch of the same guardian
            self._grant = grant
            self._grant_t = now
        return True

    def _owned_locked(self, now):
        if not self.enabled:
            return True
        g = self._grant
        if g is None or now - self._grant_t > self.timeout_s:
            return False
        return g['owner'] == self.name

    def owned(self, now):
        with self._lock:
            return self._owned_locked(now)

    @property
    def owner(self):
        with self._lock:
            return None if self._grant is None else self._grant['owner']

    @property
    def epoch(self):
        with self._lock:
            return None if self._grant is None else self._grant['epoch']

    def update(self, now):
        """REVOKED once on owned->not owned; ACQUIRED on acquire or token change."""
        if not self.enabled:
            return None
        with self._lock:
            owned = self._owned_locked(now)
            token = None
            if owned:
                token = (self._grant['epoch'], self._grant['guardian'])
            event = None
            if self._was_owned and not owned:
                event = REVOKED
            elif owned and (not self._was_owned or token != self._token):
                event = ACQUIRED
            self._was_owned = owned
            self._token = token
            return event


class EstopReassert:
    """StopMove on the engage edge, then at most max_reasserts spaced >= period_s."""

    def __init__(self, period_s=1.0, max_reasserts=3):
        self.period_s = period_s
        self.max_reasserts = max_reasserts
        self._engaged = False
        self._last = 0.0
        self._count = 0

    def on_true(self, now):
        """True if a StopMove should be sent for this /come_here/estop True."""
        if not self._engaged:
            self._engaged = True
            self._last = now
            self._count = 0
            return True
        if self._count < self.max_reasserts and now - self._last >= self.period_s:
            self._count += 1
            self._last = now
            return True
        return False

    def on_false(self):
        self._engaged = False
        self._count = 0
