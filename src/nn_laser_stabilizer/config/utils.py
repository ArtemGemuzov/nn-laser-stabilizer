from pathlib import Path


CONFIG_DIR_NAME = "configs"


def find_project_root() -> Path:
    """
    Находит корень проекта, поднимаясь от текущего файла вверх по директориям.
    Ищет маркеры: pyproject.toml и папку с конфигами.
    
    Raises:
        FileNotFoundError: Если корень проекта не найден.
    """
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "pyproject.toml").exists() and (parent / CONFIG_DIR_NAME).is_dir():
            return parent
    
    raise FileNotFoundError(
        f"Project root not found. Searched from: {current_dir}. "
        f"Looking for 'pyproject.toml' file and '{CONFIG_DIR_NAME}/' directory."
    )


def get_configs_dir() -> Path:
    """
    Возвращает абсолютный путь к папке configs относительно корня проекта.
    
    Returns:
        Path: Абсолютный путь к директории с конфигами.
        
    Raises:
        FileNotFoundError: Если папка configs не найдена.
    """
    project_root = find_project_root()
    configs_dir = (project_root / CONFIG_DIR_NAME).resolve()
    
    if not configs_dir.exists():
        raise FileNotFoundError(
            f"Configs directory not found: {configs_dir}. "
            f"Project root detected as: {project_root}"
        )
    
    return configs_dir
