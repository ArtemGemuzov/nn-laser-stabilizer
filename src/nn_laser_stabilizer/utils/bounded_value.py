from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T", int, float)


class BoundedValue(Generic[T]):
    def __init__(self, min_val: T, max_val: T, initial: T = 0):
        self._min = min_val
        self._max = max_val
        self._value: T = self._clip(initial)

    def _clip(self, x: T) -> T:
        return type(x)(np.clip(x, self._min, self._max))

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, value: T) -> None:
        self._value = self._clip(value)

    def add(self, delta: T) -> tuple[T, bool]:
        raw = self._value + delta
        self._value = self._clip(raw)
        clipped = raw != self._value
        return self._value, clipped
