import json
import math
from collections import deque
from typing import Optional

import numpy as np

from nn_laser_stabilizer.utils.logger import Logger


class ARXPlantBackend:
    """PlantBackend на основе ARX-модели, идентифицированной по экспериментальным данным.

    Модель:
        pv(t) = sum(a[i] * pv(t-i-1)) + sum(b[j] * co(t-j)) + c0 + disturbance(t)

    Возмущение — сумма синусоид с рандомизированными фазами + белый шум.
    """

    LOG_SOURCE = "arx_plant"

    def __init__(
        self,
        *,
        setpoint: int,
        a: list[float],
        b: list[float],
        c0: float,
        disturbances: list[tuple[float, float]],
        noise_std: float,
        dt: float,
        setpoint_override_probability: float = 0.0,
        pv_min: float = 0.0,
        pv_max: float = 1023.0,
        logger: Optional[Logger] = None,
    ):
        if not 0.0 <= setpoint_override_probability <= 1.0:
            raise ValueError(
                "setpoint_override_probability must be in [0, 1], "
                f"got {setpoint_override_probability}"
            )
        if pv_min > pv_max:
            raise ValueError(
                f"pv_min must be <= pv_max, got pv_min={pv_min}, pv_max={pv_max}"
            )

        self._setpoint = setpoint
        self._a = list(a)
        self._b = list(b)
        self._c0 = c0
        self._disturbances = list(disturbances)
        self._noise_std = noise_std
        self._dt = dt
        self._setpoint_override_probability = setpoint_override_probability
        self._pv_min = pv_min
        self._pv_max = pv_max
        self._logger = logger

        self._na = len(a)
        self._nb = len(b)

        self._pv_history: deque[float] = deque(maxlen=self._na)
        self._co_history: deque[int] = deque(maxlen=max(self._nb - 1, 1))
        self._disturbance_phases: list[float] = []
        self._step = 0

    @property
    def setpoint(self) -> int:
        return self._setpoint

    def _clip_pv(self, pv: float) -> float:
        return float(np.clip(pv, self._pv_min, self._pv_max))

    def reset(self) -> None:
        initial_pv = self._clip_pv(float(self._setpoint))
        self._pv_history = deque([initial_pv] * self._na, maxlen=self._na)
        self._co_history = deque([0] * max(self._nb - 1, 1), maxlen=max(self._nb - 1, 1))
        self._step = 0
        self._disturbance_phases = [
            np.random.uniform(0, 2 * math.pi)
            for _ in self._disturbances
        ]

    def exchange(self, control_output: int) -> int:
        self._step += 1
        time = self._step * self._dt

        disturbance = sum(
            amp * math.sin(2 * math.pi * freq * time + phi)
            for (freq, amp), phi in zip(self._disturbances, self._disturbance_phases)
        )
        if self._noise_std > 0:
            disturbance += np.random.normal(0, self._noise_std)

        pv = self._c0 + disturbance
        for i, ai in enumerate(self._a):
            pv += ai * self._pv_history[-(i + 1)]
        pv += self._b[0] * control_output
        for j in range(1, self._nb):
            pv += self._b[j] * self._co_history[-j]

        pv = self._clip_pv(pv)

        self._pv_history.append(pv)
        self._co_history.append(control_output)

        process_variable = int(round(pv))
        setpoint_override_triggered = (
            self._setpoint_override_probability > 0.0
            and np.random.random() < self._setpoint_override_probability
        )
        if setpoint_override_triggered:
            process_variable = int(round(self._clip_pv(float(self._setpoint))))

        if self._logger is not None:
            self._logger.log(json.dumps({
                "source": self.LOG_SOURCE,
                "event": "exchange",
                "step": self._step,
                "control_output": control_output,
                "process_variable": process_variable,
                "setpoint_override_triggered": setpoint_override_triggered,
                "disturbance": round(disturbance, 4),
                "pv_raw": round(pv, 4),
            }))

        return process_variable

    def close(self) -> None:
        pass
