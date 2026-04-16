"""Regression tests for mock-mode import safety.

Mock-mode launches must not require the ``ultralytics`` package just to
import ``come_here_perception.perception_node``. The YOLO provider is
gated behind the ``use_mock=false`` branch and must be imported lazily.
"""

import builtins
import sys

import pytest


@pytest.fixture
def ultralytics_unavailable(monkeypatch):
    """Simulate ultralytics being absent from the environment."""
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split('.')[0]
        if root == 'ultralytics':
            raise ImportError(f'ultralytics simulated unavailable: {name}')
        return original_import(name, globals, locals, fromlist, level)

    # Drop cached modules so the patched __import__ is actually hit when
    # perception_node is re-imported.
    for mod_name in list(sys.modules):
        if mod_name.startswith('ultralytics'):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
        if mod_name.startswith('come_here_perception.yolo_person_detector'):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
        if mod_name == 'come_here_perception.perception_node':
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    yield


def test_perception_node_importable_without_ultralytics(ultralytics_unavailable):
    """perception_node.py must import even when ultralytics is missing."""
    import come_here_perception.perception_node as perception_node

    assert hasattr(perception_node, 'PerceptionNode')
    assert hasattr(perception_node, 'main')
