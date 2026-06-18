"""Доменный класс ``NeuralController`` — прямое управление фазовращателем сетью.

Специализированный ``Experiment``: добавляет типизированные неймспейсы
``plant`` / ``interaction`` / ``train`` поверх канонических таблиц. Параметры
прогона достаются из канонического конфига в свой ``NeuralControllerParams`` —
наружу конфиг не отдаётся, его значения только питают производные сигналы.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property

from nn_laser_stabilizer.analysis.experiments.experiment import Experiment
from nn_laser_stabilizer.analysis.experiments.format import CanonicalConfig
from nn_laser_stabilizer.analysis.experiments.namespaces import (
    Checkpoints, Interaction, Plant, Train,
)


@dataclass(frozen=True)
class NeuralControllerParams:
    """Параметры прогона, нужные именно нейроконтроллеру (из канон-конфига)."""

    setpoint: float           # физ. единицы (/10 спрятано в адаптере)
    error_factor: float
    max_delta: float
    control_range: float      # control_max − control_min, для денорм. CO-каналов
    pid_steps: int            # граница фазы ПИД-исследования (0, если exploration нет)
    target_entropy: float | None  # для восстановления entropy там, где её не логировали
    gamma: float | None       # коэффициент дисконтирования
    observe: dict             # флаги состава вектора наблюдений (observe_*)

    @classmethod
    def from_config(cls, cfg: CanonicalConfig) -> "NeuralControllerParams":
        return cls(
            setpoint=cfg.get("setpoint"),
            error_factor=cfg.get("error_factor"),
            max_delta=cfg.get("max_delta"),
            control_range=cfg.get("control_max") - cfg.get("control_min"),
            pid_steps=cfg.get("exploration.steps", 0),
            target_entropy=cfg.get("target_entropy", None),
            gamma=cfg.get("gamma", None),
            observe={
                flag: cfg.get(flag, False)
                for flag in (
                    "observe_prev_error",
                    "observe_prev_prev_error",
                    "observe_prev_control_output",
                    "observe_prev_prev_control_output",
                )
            },
        )


class NeuralController(Experiment):
    # Линия прямого управления сетью: neural_controller / neural-controller(-vN).
    _NAME_RE = re.compile(r"neural[-_]controller")

    def _validate(self) -> None:
        if not self._NAME_RE.search(self.name):
            raise TypeError(
                f"{self._dir.name} — не нейроконтроллер (имя линии: {self.name})"
            )
        if not self.has_file("collector.jsonl", "env.jsonl"):
            raise TypeError(
                f"{self._dir.name}: нет ожидаемого лога траектории "
                f"(collector.jsonl / env.jsonl)"
            )

    @cached_property
    def _params(self) -> NeuralControllerParams:
        return NeuralControllerParams.from_config(self._fmt.config)

    @property
    def gamma(self) -> float | None:
        """Коэффициент дисконтирования (из конфига алгоритма)."""
        return self._params.gamma

    @cached_property
    def plant(self) -> Plant:
        return Plant(self._fmt.exchange,
                     duration_seconds=self._fmt.duration_seconds,
                     setpoint=self._params.setpoint)

    @cached_property
    def interaction(self) -> Interaction:
        return Interaction(self._fmt.step,
                           duration_seconds=self._fmt.duration_seconds,
                           error_factor=self._params.error_factor,
                           max_delta=self._params.max_delta,
                           control_range=self._params.control_range,
                           pid_steps=self._params.pid_steps,
                           observe=self._params.observe)

    @cached_property
    def train(self) -> Train:
        return Train(self._fmt.train, self._fmt.evaluation,
                     duration_seconds=self._fmt.duration_seconds,
                     target_entropy=self._params.target_entropy)

    @cached_property
    def checkpoints(self) -> Checkpoints:
        return Checkpoints(self._dir, self._fmt.raw_config)
