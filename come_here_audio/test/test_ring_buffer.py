"""Unit tests for RingBuffer and LatestOnlyQueue."""

import queue
import threading
import time

import numpy as np
import pytest

from come_here_audio.ring_buffer import LatestOnlyQueue, RingBuffer


class TestRingBuffer:
    def test_basic_write_read(self):
        rb = RingBuffer(capacity=100)
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        rb.write(data)
        result = rb.read_last(3)
        np.testing.assert_array_equal(result, data)

    def test_read_more_than_written(self):
        rb = RingBuffer(capacity=100)
        rb.write(np.array([1.0, 2.0], dtype=np.float32))
        result = rb.read_last(5)
        assert len(result) == 5
        # First 3 should be zero-padded
        np.testing.assert_array_equal(result[:3], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result[3:], [1.0, 2.0])

    def test_wraparound(self):
        rb = RingBuffer(capacity=5)
        rb.write(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        rb.write(np.array([4.0, 5.0, 6.0], dtype=np.float32))
        result = rb.read_last(5)
        np.testing.assert_array_equal(result, [2.0, 3.0, 4.0, 5.0, 6.0])

    def test_overwrite_full_buffer(self):
        rb = RingBuffer(capacity=3)
        rb.write(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
        result = rb.read_last(3)
        np.testing.assert_array_equal(result, [3.0, 4.0, 5.0])

    def test_total_written(self):
        rb = RingBuffer(capacity=10)
        rb.write(np.array([1.0, 2.0], dtype=np.float32))
        assert rb.total_written == 2
        rb.write(np.array([3.0], dtype=np.float32))
        assert rb.total_written == 3

    def test_empty_write(self):
        rb = RingBuffer(capacity=10)
        rb.write(np.array([], dtype=np.float32))
        assert rb.total_written == 0

    def test_read_last_partial(self):
        rb = RingBuffer(capacity=10)
        rb.write(np.arange(10, dtype=np.float32))
        result = rb.read_last(3)
        np.testing.assert_array_equal(result, [7.0, 8.0, 9.0])

    def test_thread_safety(self):
        rb = RingBuffer(capacity=16000)
        errors = []

        def writer():
            for _ in range(100):
                rb.write(np.random.randn(160).astype(np.float32))
                time.sleep(0.001)

        def reader():
            for _ in range(100):
                data = rb.read_last(1600)
                if len(data) != 1600:
                    errors.append(f"Got length {len(data)}")
                time.sleep(0.001)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == [], f"Thread safety errors: {errors}"


class TestLatestOnlyQueue:
    def test_basic_put_get(self):
        q = LatestOnlyQueue()
        q.put("hello")
        assert q.get(timeout=1.0) == "hello"

    def test_drops_old_item(self):
        q = LatestOnlyQueue()
        q.put("old")
        q.put("new")
        assert q.get(timeout=1.0) == "new"

    def test_get_nowait_empty(self):
        q = LatestOnlyQueue()
        with pytest.raises(queue.Empty):
            q.get_nowait()

    def test_empty_check(self):
        q = LatestOnlyQueue()
        assert q.empty()
        q.put("item")
        assert not q.empty()

    def test_multiple_puts_keeps_latest(self):
        q = LatestOnlyQueue()
        for i in range(10):
            q.put(i)
        assert q.get(timeout=1.0) == 9
