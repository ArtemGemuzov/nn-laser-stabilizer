"""Базовый ``Experiment`` и идентификатор прогона.

База несёт только общую механику: резолв каталога по :class:`ExperimentId`
и дешёвые метаданные. Чтение логов, канонизация и доменные узлы — в
конкретных экспериментах (``experiments/<line>.py``), каждый самодостаточен.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nn_laser_stabilizer.utils.paths import (
    get_experiment_dir,
    get_experiment_dir_name,
    parse_experiment_dir_name,
)


@dataclass(frozen=True)
class ExperimentId:
    """Идентификатор прогона: имя линии + дата + время каталога.

    Можно собрать из полей или сразу из имени каталога:
        ExperimentId("neural_controller-v3", "2026-03-02", "10-33-18")
        ExperimentId.from_dir_name("2026-03-02_10-33-18_neural_controller-v3")
    Конструктор проверяет формат даты/времени/имени и бросает ``ValueError``.
    """

    name: str
    date: str
    time: str

    def __post_init__(self) -> None:
        dir_name = get_experiment_dir_name(
            experiment_name=self.name,
            experiment_date=self.date,
            experiment_time=self.time,
        )
        if parse_experiment_dir_name(dir_name) is None:
            raise ValueError(
                f"некорректный ExperimentId: {dir_name!r} "
                f"(ожидается <date>_<time>_<name>, напр. "
                f"'2026-03-02_10-33-18_neural_controller-v3')"
            )

    @classmethod
    def from_dir_name(cls, dir_name: str) -> "ExperimentId":
        """Собрать из имени каталога ``<date>_<time>_<name>``."""
        parsed = parse_experiment_dir_name(dir_name)
        if parsed is None:
            raise ValueError(
                f"некорректное имя каталога эксперимента: {dir_name!r}"
            )
        return cls(name=parsed.name, date=parsed.date, time=parsed.time)

    @property
    def dir(self) -> Path:
        return get_experiment_dir(
            experiment_name=self.name,
            experiment_date=self.date,
            experiment_time=self.time,
        )


class Experiment:
    """Каталог по id и дешёвые метаданные. Узлы добавляет конкретный потомок."""

    def __init__(self, id: ExperimentId):
        self._id = id
        self._dir = id.dir
        if not self._dir.exists():
            raise FileNotFoundError(self._dir)

    @property
    def id(self) -> ExperimentId:
        return self._id

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def name(self) -> str:
        """Имя эксперимента (суффикс каталога) — основной признак линии."""
        parsed = parse_experiment_dir_name(self._dir.name)
        return parsed.name if parsed is not None else self._dir.name

    def has_file(self, *names: str) -> bool:
        """Есть ли в каталоге эксперимента любой из перечисленных файлов."""
        return any((self._dir / n).exists() for n in names)
