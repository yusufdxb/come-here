"""Tests for the FaceDetector ABC and MockFaceDetector."""

import numpy as np

from come_here_perception.face_detector import (
    EMPTY_RESULT,
    FaceDetectionResult,
    MockFaceDetector,
)


def _blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_mock_default_returns_empty():
    det = MockFaceDetector()
    det.setup()
    r = det.detect(_blank_frame())
    assert r.face_present is False
    assert r.face_count == 0
    assert r.max_confidence == 0.0
    det.teardown()


def test_mock_returns_configured_result():
    expected = FaceDetectionResult(
        face_present=True,
        face_count=1,
        max_confidence=0.87,
        center_x_norm=0.42,
        center_y_norm=0.55,
    )
    det = MockFaceDetector(result=expected)
    det.setup()
    r = det.detect(_blank_frame())
    assert r == expected


def test_mock_set_result_mutates():
    det = MockFaceDetector()
    det.setup()
    assert det.detect(_blank_frame()).face_present is False
    det.set_result(FaceDetectionResult(True, 2, 0.9, 0.3, 0.6))
    r = det.detect(_blank_frame())
    assert r.face_present is True
    assert r.face_count == 2
    assert r.max_confidence == 0.9


def test_empty_result_is_centered():
    assert EMPTY_RESULT.face_present is False
    assert EMPTY_RESULT.center_x_norm == 0.5
    assert EMPTY_RESULT.center_y_norm == 0.5
