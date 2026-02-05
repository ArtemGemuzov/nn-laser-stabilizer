from dataclasses import dataclass

import numpy as np


@dataclass
class PIDConfig:
    kp: float
    ki: float
    kd: float
    dt: float
    min_output: float
    max_output: float


class PID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float,
        min_output: float,
        max_output: float,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.min_output = min_output
        self.max_output = max_output

        self._integral: float = 0.0
        self._prev_error: float = 0.0

    @classmethod
    def from_config(cls, config: PIDConfig) -> "PID":
        return cls(
            kp=config.kp,
            ki=config.ki,
            kd=config.kd,
            dt=config.dt,
            min_output=config.min_output,
            max_output=config.max_output,
        )

    def compute(self, process_variable: float, setpoint: float) -> float:
        error = process_variable - setpoint

        self._integral = np.clip(
            self._integral + error * self.dt * self.ki,
            self.min_output,
            self.max_output,
        )

        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error

        output = error * self.kp + self._integral + derivative * self.kd
        return np.clip(output, self.min_output, self.max_output)

    def __call__(self, process_variable: float, setpoint: float) -> float:
        return self.compute(process_variable, setpoint)


class IncrementalPID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float,
        min_output: float,
        max_output: float,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.min_output = min_output
        self.max_output = max_output

        self._prev_error: float = 0.0
        self._prev_prev_error: float = 0.0
        self._output: float = 0.0

    @classmethod
    def from_config(cls, config: PIDConfig) -> "IncrementalPID":
        return cls(
            kp=config.kp,
            ki=config.ki,
            kd=config.kd,
            dt=config.dt,
            min_output=config.min_output,
            max_output=config.max_output,
        )

    def compute(self, process_variable: float, setpoint: float) -> float:
        error = process_variable - setpoint

        delta_p = self.kp * (error - self._prev_error)
        delta_i = self.ki * error * self.dt
        delta_d = self.kd * (error - 2 * self._prev_error + self._prev_prev_error) / self.dt

        delta_output = delta_p + delta_i + delta_d

        self._output = np.clip(
            self._output + delta_output,
            self.min_output,
            self.max_output,
        )

        self._prev_prev_error = self._prev_error
        self._prev_error = error
        return self._output

    def __call__(self, process_variable: float, setpoint: float) -> float:
        return self.compute(process_variable, setpoint)

    @property
    def output(self) -> float:
        return self._output

    @output.setter
    def output(self, value: float) -> None:
        self._output = np.clip(value, self.min_output, self.max_output)