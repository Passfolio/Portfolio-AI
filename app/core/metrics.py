"""함수 실행 시간 추적 데코레이터."""
from __future__ import annotations

import functools
import logging
import time
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

logger = logging.getLogger(__name__)


def track_metrics(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.debug("%s 실행 시간: %.3fs", func.__qualname__, elapsed)

    return wrapper  # type: ignore[return-value]
