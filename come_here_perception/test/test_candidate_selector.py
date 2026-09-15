"""DOA-consistent caller selection (pure Python)."""

import math

from come_here_perception.candidate_selector import Candidate, largest, select


def c(deg, conf=0.9, h=0.5):
    return Candidate(math.radians(deg), 2.0, conf, h)


def test_closest_to_the_voice_wins_over_a_bigger_bystander():
    people = [c(-30, h=0.9), c(4, h=0.4), c(35, h=0.7)]
    idx, in_gate = select(people, 0.0, math.radians(25))
    assert idx == 1
    assert in_gate == [False, True, False]


def test_nobody_inside_the_gate_selects_nobody():
    idx, in_gate = select([c(-40), c(40)], 0.0, math.radians(25))
    assert idx is None and in_gate == [False, False]


def test_closed_gate_selects_nobody():
    assert select([c(0)], 0.0, 0.0)[0] is None


def test_gate_center_off_axis_after_a_short_turn():
    people = [c(0, h=0.9), c(18)]
    assert select(people, math.radians(20), math.radians(25))[0] == 1


def test_ties_break_on_confidence_then_height():
    assert select([c(5, conf=0.6), c(-5, conf=0.9)], 0.0, 0.5)[0] == 1
    assert select([c(5, h=0.3), c(-5, h=0.6)], 0.0, 0.5)[0] == 1


def test_wraps_around_pi():
    assert select([c(178)], math.radians(-178), math.radians(10))[0] == 0


def test_legacy_largest_without_gate():
    assert largest([c(-30, h=0.2), c(10, h=0.8)]) == 1
    assert largest([]) is None
