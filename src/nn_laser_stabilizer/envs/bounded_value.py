from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T", int, float)


class BoundedValue(Generic[T]):
    def __init__(self, min_val: T, max_val: T, initial: T = 0):
        self._min = min_val
        self._max = max_val
        self._value: T = self._clip(initial)

    def _clip(self, x: T) -> T:
        clipped: T = np.clip(x, self._min, self._max)
        return clipped

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, value: T) -> None:
        self._value = self._clip(value)

    def add(self, delta: T) -> T:
        self._value = self._clip(self._value + delta)
        return self._value
