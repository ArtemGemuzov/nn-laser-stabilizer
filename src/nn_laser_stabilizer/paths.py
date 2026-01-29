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