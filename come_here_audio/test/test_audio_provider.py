"""Unit tests for audio direction provider abstraction."""

from come_here_audio.audio_direction_provider import DirectionEstimate
from come_here_audio.mock_audio_provider import MockAudioProvider


def test_mock_provider_returns_estimate():
    provider = MockAudioProvider(fixed_azimuth_rad=0.5)
    provider.setup()
    result = provider.get_direction()
    assert result is not None
    assert isinstance(result, DirectionEstimate)
    assert 0.0 <= result.confidence <= 1.0
    provider.teardown()


def test_mock_provider_inactive_returns_none():
    provider = MockAudioProvider()
    result = provider.get_direction()
    assert result is None


def test_mock_wake_phrase_detector():
    from come_here_audio.wake_phrase_detector import MockWakePhraseDetector

    det = MockWakePhraseDetector()
    det.setup()
    assert det.check() is None
    det.set_triggered(True)
    result = det.check()
    assert result is not None
    assert result.phrase == "come here"
    # Should auto-reset
    assert det.check() is None
    det.teardown()
