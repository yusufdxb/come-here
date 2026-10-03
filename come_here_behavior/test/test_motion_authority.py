"""Motion authority gate (pure) and bridge wiring tests."""

import json

import pytest

from come_here_behavior.motion_authority import (
    ACQUIRED, REVOKED, AuthorityGate, EstopReassert, parse_grant,
)


def grant(owner='come_here', epoch=1, guardian='g1', state='RUN', mode='mcf', **kw):
    d = {'v': 1, 'owner': owner, 'epoch': epoch, 'guardian': guardian,
         'state': state, 'mode': mode}
    d.update(kw)
    return json.dumps(d)


def test_owned_with_fresh_grant():
    g = AuthorityGate('come_here', 0.3)
    assert not g.owned(0.0)
    g.on_grant(grant(), 1.0)
    assert g.owned(1.2)
    assert g.update(1.2) == ACQUIRED
    assert g.update(1.25) is None


def test_stale_grant_not_owned_and_revokes_once():
    g = AuthorityGate('come_here', 0.3)
    g.on_grant(grant(), 1.0)
    g.update(1.0)
    assert g.owned(1.25)
    assert not g.owned(1.31)
    assert g.update(1.31) == REVOKED
    assert g.update(1.5) is None


@pytest.mark.parametrize('owner', ['nav2', None])
def test_other_owner_or_null_revokes_once(owner):
    g = AuthorityGate('come_here', 0.3)
    g.on_grant(grant(), 1.0)
    g.update(1.0)
    g.on_grant(grant(owner=owner), 1.1)
    assert not g.owned(1.1)
    assert g.update(1.1) == REVOKED
    g.on_grant(grant(owner=owner), 1.2)
    assert g.update(1.2) is None


def test_epoch_or_guardian_change_reacquires():
    g = AuthorityGate('come_here', 0.3)
    g.on_grant(grant(epoch=1), 1.0)
    assert g.update(1.0) == ACQUIRED
    g.on_grant(grant(epoch=2), 1.1)
    assert g.update(1.1) == ACQUIRED
    g.on_grant(grant(epoch=2, guardian='g2'), 1.2)
    assert g.update(1.2) == ACQUIRED
    g.on_grant(grant(epoch=2, guardian='g2'), 1.25)
    assert g.update(1.25) is None


@pytest.mark.parametrize('raw', [
    'not json', '[]', '{}', None, 5,
    grant(v=2), grant(v='1'), grant(epoch='1'), grant(epoch=True),
    grant(owner=7), grant(guardian=3), grant(state=None), grant(mode=1),
])
def test_malformed_grant_ignored(raw):
    assert parse_grant(raw) is None
    g = AuthorityGate('come_here', 0.3)
    assert g.on_grant(raw, 1.0) is False
    assert not g.owned(1.0)
    g.on_grant(grant(), 1.0)
    g.on_grant(raw, 1.1)  # must not refresh or revoke
    assert g.owned(1.2) and not g.owned(1.4)


def test_legacy_always_owned_no_events():
    g = AuthorityGate('come_here', 0.3, enabled=False)
    assert g.owned(0.0) and g.owned(1e9)
    assert g.update(0.0) is None
    g.on_grant(grant(owner='nav2'), 1.0)
    assert g.owned(5.0) and g.update(5.0) is None


def test_estop_reassert_policy():
    r = EstopReassert()
    sent = [r.on_true(i * (5.0 / 30)) for i in range(30)]
    assert sent[0] is True
    assert sum(sent) <= 4 and sum(sent[1:]) <= 3
    assert r.on_true(10.0) is False  # budget spent
    r.on_false()
    assert r.on_true(11.0) is True


# -- bridge wiring (needs unitree_api) --

try:
    from come_here_behavior.go2_bridge_node import Go2BridgeNode  # noqa: F401
    from test.test_bridge_safety import (
        STOP_MOVE_API_ID, MOVE_API_ID, _api_ids, _bool, _make_node, _mark, _moves,
        _rclpy_runtime, _vel,
    )
    _BRIDGE = True
except ImportError:
    _BRIDGE = False

bridge = pytest.mark.skipif(not _BRIDGE, reason='bridge not importable')
SIT, STAND = 1005, 1002


def _grant(node, **kw):
    node._authority.on_grant(grant(**kw), node._now())
    node._authority_poll()


@pytest.fixture
def anode():
    n = _make_node(motion_authority_topic='/auth', grant_timeout_s=0.3)
    yield n
    n.destroy_node()


@bridge
def test_estop_repeated_true_bounded():
    n = _make_node()
    for i in range(30):
        n._now.t = 100.0 + i * (5.0 / 30)
        n._estop_cb(_bool(True))
    stops = _api_ids(n).count(STOP_MOVE_API_ID)
    assert 1 <= stops <= 4
    n._estop_cb(_bool(False))
    m = _mark(n)
    n._estop_cb(_bool(True))
    assert _api_ids(n, m) == [STOP_MOVE_API_ID]
    n.destroy_node()


@bridge
def test_no_move_without_grant_and_stop_still_sent(anode):
    n = anode
    n._velocity_cb(_vel(0.6, 0.0))
    n._velocity_tick()
    assert _moves(n) == []
    assert n._authority_dropped >= 1
    n._estop_cb(_bool(True))
    assert STOP_MOVE_API_ID in _api_ids(n)


@bridge
def test_grant_allows_move_then_revoke_one_stop_and_needs_fresh_command(anode):
    n = anode
    _grant(n)
    n._velocity_cb(_vel(0.6, 0.0))
    assert len(_moves(n)) == 1
    n._now.t += 0.1
    _grant(n, owner='nav2')
    assert _api_ids(n).count(STOP_MOVE_API_ID) == 1
    m = _mark(n)
    for _ in range(5):
        n._now.t += 0.05
        n._velocity_tick()
    assert _api_ids(n, m) == []
    # Grant returns: still no Move until a fresh command.
    n._now.t += 0.05
    _grant(n)
    n._velocity_tick()
    assert _moves(n, m) == []
    n._velocity_cb(_vel(0.6, 0.0))
    assert len(_moves(n, m)) == 1


@bridge
def test_stale_grant_revokes_via_tick(anode):
    n = anode
    _grant(n)
    n._velocity_cb(_vel(0.6, 0.0))
    n._now.t += 0.5
    n._velocity_tick()
    assert _api_ids(n).count(STOP_MOVE_API_ID) == 1
    assert len(_moves(n)) == 1


@bridge
def test_sit_and_stand_dropped_when_not_owned(anode):
    n = anode
    n._stand_cb(_bool(True))
    n._deferred_sport_call(SIT, 0.0, 'sit')
    assert SIT not in _api_ids(n) and STAND not in _api_ids(n)
    _grant(n)
    n._stand_cb(_bool(True))
    n._deferred_sport_call(SIT, 0.0, 'sit')
    assert STAND in _api_ids(n) and SIT in _api_ids(n)


@bridge
def test_legacy_default_unchanged():
    n = _make_node()
    n._velocity_cb(_vel(0.6, 0.0))
    assert len(_moves(n)) == 1
    st = json.loads(json.dumps({'e': n._authority.enabled}))
    assert st['e'] is False
    n.destroy_node()
