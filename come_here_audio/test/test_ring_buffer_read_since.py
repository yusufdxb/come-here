"""RingBuffer.read_since: contiguous reads with overrun detection."""

import numpy as np

from come_here_audio.ring_buffer import RingBuffer


def test_returns_only_new_samples_and_the_end_position():
    rb = RingBuffer(capacity=10)
    rb.write(np.arange(4, dtype=np.float32))
    samples, end = rb.read_since(0)
    assert end == 4 and list(samples) == [0, 1, 2, 3]
    rb.write(np.arange(4, 7, dtype=np.float32))
    samples, end = rb.read_since(end)
    assert end == 7 and list(samples) == [4, 5, 6]


def test_nothing_new_returns_empty():
    rb = RingBuffer(capacity=10)
    rb.write(np.arange(3, dtype=np.float32))
    samples, end = rb.read_since(3)
    assert end == 3 and len(samples) == 0


def test_wraparound_is_contiguous():
    rb = RingBuffer(capacity=5)
    rb.write(np.arange(4, dtype=np.float32))
    _, end = rb.read_since(0)
    rb.write(np.arange(4, 7, dtype=np.float32))
    samples, end = rb.read_since(end)
    assert list(samples) == [4, 5, 6]


def test_overrun_is_detectable():
    rb = RingBuffer(capacity=5)
    rb.write(np.arange(12, dtype=np.float32))
    samples, end = rb.read_since(0)
    assert end == 12 and len(samples) == 5
    assert end - len(samples) > 0  # the reader lost samples 0..6
    assert list(samples) == [7, 8, 9, 10, 11]
