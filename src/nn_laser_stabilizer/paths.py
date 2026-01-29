from typing import Optional
import os
from pathlib import Path


def find_project_root() -> Path:
    """
    Находит корень проекта, поднимаясь от текущего файла вверх по директориям.
    Ищет маркер: pyproject.toml.

    Raises:
        FileNotFoundError: Если корень проекта не найден.
    """
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent

    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    raise FileNotFoundError(
        f"Project root not found. Searched from: {current_dir}. "
        f"Looking for 'pyproject.toml' file."
    )


class WorkingDirectoryContext:
    def __init__(self, target_dir: Path):
        self.target_dir = Path(target_dir)
        self._old_cwd: Optional[Path] = None

    def __enter__(self) -> "WorkingDirectoryContext":
        self._old_cwd = Path.cwd()
        os.chdir(self.target_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._old_cwd is not None:
            os.chdir(self._old_cwd)
        return False