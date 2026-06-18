"""Типизированный доступ к экспериментам: ``repo.get(...).expect(NeuralController)``.

Пилот: репозиторий → базовый ``Experiment`` → специализация ``NeuralController``
(неймспейсы ``plant`` / ``interaction`` / ``train`` поверх канонических таблиц,
которые адаптер формата строит из лог-файлов поколения).
"""

from nn_laser_stabilizer.analysis.experiments.experiment import (
    Experiment, ExperimentRepository,
)
from nn_laser_stabilizer.analysis.experiments.neural_controller import (
    NeuralController, NeuralControllerParams,
)

__all__ = [
    "ExperimentRepository",
    "Experiment",
    "NeuralController",
    "NeuralControllerParams",
]
