from typing import Optional
import os
from pathlib import Path


EXPERIMENTS_DIR_NAME = "experiments"
CONFIGS_DIR_NAME = "configs"
DATA_DIR_NAME = "data"
RESOURCES_DIR_NAME = "resources"


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


def get_dir(dir_name: str) -> Path:
    project_root = find_project_root()
    dir_path = (project_root / dir_name).resolve()
    if not dir_path.exists():
        raise FileNotFoundError(
            f"{dir_name} directory not found: {dir_path}. "
            f"Project root detected as: {project_root}"
        )
    return dir_path


def get_or_create_dir(dir_name: str) -> Path:
    project_root = find_project_root()
    dir_path = (project_root / dir_name).resolve()
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_configs_dir() -> Path:
    return get_dir(CONFIGS_DIR_NAME)


def get_data_dir() -> Path:
    return get_dir(DATA_DIR_NAME)


def get_resources_dir() -> Path:
    return get_dir(RESOURCES_DIR_NAME)


def get_experiments_dir() -> Path:
    return get_dir(EXPERIMENTS_DIR_NAME)


def get_or_create_experiments_dir() -> Path:
    return get_or_create_dir(EXPERIMENTS_DIR_NAME)


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