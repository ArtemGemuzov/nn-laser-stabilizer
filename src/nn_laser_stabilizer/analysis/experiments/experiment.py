"""Базовый ``Experiment`` и репозиторий.

``Experiment`` — общая механика (каталог, определение формата, дешёвые
метаданные, ``expect``). Доменные классы — специализированные ``Experiment``
над тем же каталогом; ``expect`` повышает базовый эксперимент до специализации,
переиспользуя уже определённый формат. Валидация — инвариант конструктора
(хук ``_validate``), поэтому отдельного ``matches`` нет.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from nn_laser_stabilizer.analysis.experiments.format import detect_format
from nn_laser_stabilizer.utils.paths import get_experiments_dir, parse_experiment_dir_name

T = TypeVar("T", bound="Experiment")


class Experiment:
    def __init__(self, dir: Path, fmt=None):
        self._dir = Path(dir)
        self._fmt = fmt if fmt is not None else detect_format(self._dir)
        self._validate()

    def _validate(self) -> None:
        """Хук валидации; база ограничений не накладывает, специализация — да."""

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

    def expect(self, domain: type[T]) -> T:
        return domain(self._dir, self._fmt)   # конструктор сам бросит, если не подходит


class ExperimentRepository:
    """Находит прогоны в ``experiments/`` и отдаёт сырой ``Experiment``."""

    def __init__(self, root: Path | None = None):
        self._root = Path(root) if root is not None else get_experiments_dir()

    def get(self, *, name: str, date: str, time: str) -> Experiment:
        d = self._root / f"{date}_{time}_{name}"
        if not d.exists():
            raise FileNotFoundError(d)
        return Experiment(d)

    def all(self) -> list[Experiment]:
        out: list[Experiment] = []
        for d in sorted(self._root.iterdir()):
            if d.is_dir() and parse_experiment_dir_name(d.name) is not None:
                out.append(Experiment(d))
        return out
