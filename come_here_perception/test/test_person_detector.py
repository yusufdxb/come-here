from come_here_perception.person_detector import MockPersonDetector


def test_mock_detector_default_no_person():
    det = MockPersonDetector()
    det.setup()
    result = det.detect()
    assert not result.detected
    assert result.confidence == 0.0
    det.teardown()


def test_mock_detector_set_detected():
    det = MockPersonDetector()
    det.setup()
    det.set_detected(True, bearing_rad=0.3, distance_m=1.5)
    result = det.detect()
    assert result.detected
    assert result.bearing_rad == 0.3
    assert result.distance_m == 1.5
    det.teardown()
