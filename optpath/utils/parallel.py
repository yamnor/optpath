"""Parallel execution helpers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def thread_env(threads: int) -> Iterator[None]:
    """Temporarily set common thread-count env vars."""
    keys = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
    previous = {key: os.environ.get(key) for key in keys}
    try:
      for key in keys:
        os.environ[key] = str(threads)
      yield
    finally:
      for key, value in previous.items():
        if value is None:
          os.environ.pop(key, None)
        else:
          os.environ[key] = value

