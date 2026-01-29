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


def get_or_create_experiments_dir() -> Path:
    project_root = find_project_root()
    experiments_dir = (project_root / EXPERIMENTS_DIR_NAME).resolve()
    experiments_dir.mkdir(parents=True, exist_ok=True)
    return experiments_dir


def get_experiments_dir() -> Path:
    project_root = find_project_root()
    experiments_dir = (project_root / EXPERIMENTS_DIR_NAME).resolve()
    
    if not experiments_dir.exists():
        raise FileNotFoundError(
            f"Experiments directory not found: {experiments_dir}. "
            f"Project root detected as: {project_root}"
        )
    
    return experiments_dir


def get_configs_dir() -> Path:
    project_root = find_project_root()
    configs_dir = (project_root / CONFIGS_DIR_NAME).resolve()
    
    if not configs_dir.exists():
        raise FileNotFoundError(
            f"Configs directory not found: {configs_dir}. "
            f"Project root detected as: {project_root}"
        )
    
    return configs_dir


def get_data_dir() -> Path:
    project_root = find_project_root()
    data_dir = (project_root / DATA_DIR_NAME).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            f"Project root detected as: {project_root}"
        )
    return data_dir


def get_resources_dir() -> Path:
    project_root = find_project_root()
    resources_dir = (project_root / RESOURCES_DIR_NAME).resolve()
    
    if not resources_dir.exists():
        raise FileNotFoundError(
            f"Resources directory not found: {resources_dir}. "
            f"Project root detected as: {project_root}"
        )
    
    return resources_dir


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