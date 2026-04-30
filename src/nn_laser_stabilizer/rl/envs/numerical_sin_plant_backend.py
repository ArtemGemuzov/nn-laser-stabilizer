import json
import math
from typing import Optional

import numpy as np

from nn_laser_stabilizer.utils.logger import Logger


class NumericalSinPlantBackend:
    """Численная модель установки с синусоидальной характеристикой.

    Модель:
        pv = 1500 + 500 * sin(control_output / 100 * 2 * pi) + random(-2, 2)
    """

    LOG_SOURCE = "numerical_sin_plant"

    def __init__(
        self,
        *,
        setpoint: int,
        noise_min: float = -2.0,
        noise_max: float = 2.0,
        pv_min: Optional[float] = None,
        pv_max: Optional[float] = None,
        logger: Optional[Logger] = None,
    ):
        if noise_min > noise_max:
            raise ValueError(
                f"noise_min must be <= noise_max, got noise_min={noise_min}, noise_max={noise_max}"
            )
        if pv_min is not None and pv_max is not None and pv_min > pv_max:
            raise ValueError(
                f"pv_min must be <= pv_max, got pv_min={pv_min}, pv_max={pv_max}"
            )

        self._setpoint = int(setpoint)
        self._noise_min = float(noise_min)
        self._noise_max = float(noise_max)
        self._pv_min = pv_min
        self._pv_max = pv_max
        self._logger = logger
        self._step = 0

    @property
    def setpoint(self) -> int:
        return self._setpoint

    def reset(self) -> None:
        self._step = 0

    def _clip_pv(self, value: float) -> float:
        if self._pv_min is not None:
            value = max(self._pv_min, value)
        if self._pv_max is not None:
            value = min(self._pv_max, value)
        return float(value)

    def exchange(self, control_output: int) -> int:
        self._step += 1

        signal = 1500.0 + 500.0 * math.sin((float(control_output) / 100.0) * 2.0 * math.pi)
        noise = float(np.random.uniform(self._noise_min, self._noise_max))
        pv_raw = signal + noise
        pv = self._clip_pv(pv_raw)
        process_variable = int(round(pv))

        if self._logger is not None:
            self._logger.log(json.dumps({
                "source": self.LOG_SOURCE,
                "event": "exchange",
                "step": self._step,
                "control_output": int(control_output),
                "process_variable": process_variable,
                "signal": round(signal, 4),
                "noise": round(noise, 4),
                "pv_raw": round(pv_raw, 4),
            }))

        return process_variable

    def close(self) -> None:
        pass
