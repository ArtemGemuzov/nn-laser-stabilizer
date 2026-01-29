from pathlib import Path


from nn_laser_stabilizer.paths import find_project_root


CONFIG_DIR_NAME = "configs"


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
