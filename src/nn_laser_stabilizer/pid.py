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
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._dt = dt
        self._ki_dt = ki * dt
        self._kd_over_dt = kd / dt
        self.min_output = min_output
        self.max_output = max_output

        self._integral: float = 0.0
        self._prev_error: float = 0.0

    @property
    def kp(self) -> float:
        return self._kp

    @property
    def ki(self) -> float:
        return self._ki

    @property
    def kd(self) -> float:
        return self._kd

    @property
    def dt(self) -> float:
        return self._dt

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
            self._integral + error * self._ki_dt,
            self.min_output,
            self.max_output,
        )

        derivative = (error - self._prev_error) * self._kd_over_dt
        self._prev_error = error

        output = error * self._kp + self._integral + derivative
        return np.clip(output, self.min_output, self.max_output)

    def __call__(self, process_variable: float, setpoint: float) -> float:
        return self.compute(process_variable, setpoint)


class PIDDelta:
    """Вычисляет приращение (дельту) инкрементального ПИД-регулятора.

    Предвычисленные коэффициенты позволяют свести вычисление к трём умножениям:
        delta = c0 * error + c1 * prev_error + c2 * prev_prev_error
    где:
        c0 = kp + ki * dt + kd / dt
        c1 = -(kp + 2 * kd / dt)
        c2 = kd / dt
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float,
    ) -> None:
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._dt = dt
        
        self._c0 = kp + ki * dt + kd / dt
        self._c1 = -(kp + 2.0 * kd / dt)
        self._c2 = kd / dt

        self._prev_error: float = 0.0
        self._prev_prev_error: float = 0.0

    @property
    def kp(self) -> float:
        return self._kp

    @property
    def ki(self) -> float:
        return self._ki

    @property
    def kd(self) -> float:
        return self._kd

    @property
    def dt(self) -> float:
        return self._dt

    @classmethod
    def from_config(cls, config: PIDConfig) -> "PIDDelta":
        return cls(
            kp=config.kp,
            ki=config.ki,
            kd=config.kd,
            dt=config.dt,
        )

    def compute(self, process_variable: float, setpoint: float) -> float:
        error = process_variable - setpoint

        delta = (
            self._c0 * error
            + self._c1 * self._prev_error
            + self._c2 * self._prev_prev_error
        )

        self._prev_prev_error = self._prev_error
        self._prev_error = error
        return delta

    def __call__(self, process_variable: float, setpoint: float) -> float:
        return self.compute(process_variable, setpoint)

    def reset(self) -> None:
        self._prev_error = 0.0
        self._prev_prev_error = 0.0


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
        self._delta = PIDDelta(kp=kp, ki=ki, kd=kd, dt=dt)
        self.min_output = min_output
        self.max_output = max_output

        self._output: float = 0.0

    @property
    def kp(self) -> float:
        return self._delta.kp

    @property
    def ki(self) -> float:
        return self._delta.ki

    @property
    def kd(self) -> float:
        return self._delta.kd

    @property
    def dt(self) -> float:
        return self._delta.dt

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
        delta = self._delta.compute(process_variable, setpoint)

        self._output = np.clip(
            self._output + delta,
            self.min_output,
            self.max_output,
        )

        return self._output

    def __call__(self, process_variable: float, setpoint: float) -> float:
        return self.compute(process_variable, setpoint)

    @property
    def output(self) -> float:
        return self._output

    @output.setter
    def output(self, value: float) -> None:
        self._output = np.clip(value, self.min_output, self.max_output)