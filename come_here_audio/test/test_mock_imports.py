"""Regression tests for mock-mode import safety.

Mock-mode launches must not require the ``usb`` (pyusb) package just to
import ``come_here_audio.audio_node``. The ReSpeaker USB HID stack is
gated behind the ``use_mock=false`` branch and must be imported lazily.
"""

import builtins
import sys

import pytest


_HARDWARE_MODULES = ('usb', 'usb.core', 'usb.util')


@pytest.fixture
def pyusb_unavailable(monkeypatch):
    """Simulate pyusb being absent from the environment."""
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split('.')[0]
        if root == 'usb':
            raise ImportError(f'pyusb simulated unavailable: {name}')
        return original_import(name, globals, locals, fromlist, level)

    # Drop cached hardware modules so the patched __import__ is actually hit.
    for mod_name in list(sys.modules):
        if mod_name.startswith('usb'):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
        if mod_name.startswith('come_here_audio.respeaker_doa_provider'):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
        if mod_name == 'come_here_audio.audio_node':
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    yield


def test_audio_node_importable_without_pyusb(pyusb_unavailable):
    """audio_node.py must import even when pyusb is missing."""
    import come_here_audio.audio_node as audio_node

    assert hasattr(audio_node, 'AudioNode')
    assert hasattr(audio_node, 'main')
