from typing import NamedTuple, Optional
import os
import re
from pathlib import Path


EXPERIMENTS_DIR_NAME = "experiments"
CONFIGS_DIR_NAME = "configs"
DATA_DIR_NAME = "data"
RESOURCES_DIR_NAME = "resources"
DOCS_DIR_NAME = "docs"

# Имя директории эксперимента: "<date>_<time>_<name>",
# например "2026-03-02_10-33-18_neural_controller-v3".
EXPERIMENT_DIR_NAME_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_(?P<name>.+)"
)


class ExperimentDirName(NamedTuple):
    date: str
    time: str
    name: str


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


def get_docs_dir() -> Path:
    return get_dir(DOCS_DIR_NAME)


def get_experiments_dir() -> Path:
    return get_dir(EXPERIMENTS_DIR_NAME)


def get_or_create_experiments_dir() -> Path:
    return get_or_create_dir(EXPERIMENTS_DIR_NAME)


def get_experiment_dir_name(
    *,
    experiment_name: str,
    experiment_date: str,
    experiment_time: str,
) -> str:
    return f"{experiment_date}_{experiment_time}_{experiment_name}"


def parse_experiment_dir_name(dir_name: str) -> Optional[ExperimentDirName]:
    match = EXPERIMENT_DIR_NAME_RE.fullmatch(dir_name)
    if match is None:
        return None
    return ExperimentDirName(
        date=match["date"],
        time=match["time"],
        name=match["name"],
    )


def get_experiment_dir(
    *,
    experiment_name: str,
    experiment_date: str,
    experiment_time: str,
) -> Path:
    experiments_dir = get_experiments_dir()
    experiment_dir_name = get_experiment_dir_name(
        experiment_name=experiment_name, 
        experiment_date=experiment_date, 
        experiment_time=experiment_time
    )
    return experiments_dir / experiment_dir_name


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