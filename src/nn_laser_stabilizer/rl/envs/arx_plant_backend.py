import json
import math
from typing import Optional

import numpy as np

from nn_laser_stabilizer.logger import Logger


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
        pv_min: float = 0.0,
        pv_max: float = 1023.0,
        logger: Optional[Logger] = None,
    ):
        self._setpoint = setpoint
        self._a = list(a)
        self._b = list(b)
        self._c0 = c0
        self._disturbances = list(disturbances)
        self._noise_std = noise_std
        self._dt = dt
        self._pv_min = pv_min
        self._pv_max = pv_max
        self._logger = logger

        self._na = len(a)
        self._nb = len(b)

        self._pv_history: list[float] = []
        self._co_history: list[int] = []
        self._disturbance_phases: list[float] = []
        self._step = 0

    @property
    def setpoint(self) -> int:
        return self._setpoint

    def reset(self) -> None:
        self._pv_history = [float(self._setpoint)] * self._na
        self._co_history = [0] * self._nb
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

        pv = float(np.clip(pv, self._pv_min, self._pv_max))

        self._pv_history.append(pv)
        self._co_history.append(control_output)

        if len(self._pv_history) > self._na + 100:
            self._pv_history = self._pv_history[-(self._na + 10):]
        if len(self._co_history) > self._nb + 100:
            self._co_history = self._co_history[-(self._nb + 10):]

        process_variable = int(round(pv))

        if self._logger is not None:
            self._logger.log(json.dumps({
                "source": self.LOG_SOURCE,
                "event": "exchange",
                "step": self._step,
                "control_output": control_output,
                "process_variable": process_variable,
                "disturbance": round(disturbance, 4),
                "pv_raw": round(pv, 4),
            }))

        return process_variable

    def close(self) -> None:
        pass
