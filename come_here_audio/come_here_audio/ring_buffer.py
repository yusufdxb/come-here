"""Thread-safe ring buffer and latest-only queue for streaming audio.

RingBuffer: numpy-backed circular buffer for continuous audio capture.
LatestOnlyQueue: single-slot queue that drops stale items on put.
"""

import queue
import threading

import numpy as np


class RingBuffer:
    """Thread-safe circular buffer backed by a numpy array.

    Designed for a single writer (audio callback) and single reader
    (segmenter thread). The lock protects write pointer updates;
    reads copy data under the same lock.

    Args:
        capacity: Maximum number of samples the buffer holds.
        dtype: Numpy dtype for the buffer (default float32).
    """

    def __init__(self, capacity: int, dtype=np.float32):
        self._buf = np.zeros(capacity, dtype=dtype)
        self._capacity = capacity
        self._write_pos = 0
        self._total_written = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_written(self) -> int:
        """Total samples written since creation (monotonic counter)."""
        return self._total_written

    def write(self, data: np.ndarray) -> None:
        """Append samples to the buffer. Wraps around if full."""
        n = len(data)
        if n == 0:
            return
        with self._lock:
            if n >= self._capacity:
                # Data larger than buffer -- keep last capacity samples
                self._buf[:] = data[-self._capacity:]
                self._write_pos = 0
                self._total_written += n
                return

            end = self._write_pos + n
            if end <= self._capacity:
                self._buf[self._write_pos:end] = data
            else:
                # Wrap around
                first = self._capacity - self._write_pos
                self._buf[self._write_pos:] = data[:first]
                self._buf[:n - first] = data[first:]

            self._write_pos = end % self._capacity
            self._total_written += n

    def read_last(self, n_samples: int) -> np.ndarray:
        """Return the last n_samples as a contiguous array.

        If fewer than n_samples have been written, the returned array
        is zero-padded at the front.
        """
        n = min(n_samples, self._capacity)
        with self._lock:
            available = min(self._total_written, self._capacity)
            if n > available:
                # Not enough data yet -- return what we have, zero-padded
                result = np.zeros(n, dtype=self._buf.dtype)
                if available > 0:
                    start = (self._write_pos - available) % self._capacity
                    if start + available <= self._capacity:
                        result[n - available:] = self._buf[start:start + available]
                    else:
                        first = self._capacity - start
                        result[n - available:n - available + first] = self._buf[start:]
                        result[n - available + first:] = self._buf[:available - first]
                return result

            start = (self._write_pos - n) % self._capacity
            if start + n <= self._capacity:
                return self._buf[start:start + n].copy()
            else:
                first = self._capacity - start
                return np.concatenate([
                    self._buf[start:],
                    self._buf[:n - first],
                ])


class LatestOnlyQueue:
    """Single-slot queue that always holds the most recent item.

    On put: if the slot is occupied, the old item is silently discarded.
    On get: blocks until an item is available (with optional timeout).
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=1)

    def put(self, item) -> None:
        """Put an item, discarding the old one if the slot is full."""
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put(item)

    def get(self, timeout: float | None = None):
        """Get the latest item. Blocks if empty."""
        return self._queue.get(timeout=timeout)

    def get_nowait(self):
        """Get without blocking. Raises queue.Empty if nothing available."""
        return self._queue.get_nowait()

    def empty(self) -> bool:
        return self._queue.empty()
