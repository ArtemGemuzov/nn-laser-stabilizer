import os


def get_hydra_runtime_output_dir() -> str:
    """
    Возвращает путь к текущей директории вывода Hydra.
    
    Если Hydra не инициализирована, возвращает текущую рабочую директорию.
    """
    try:
        from hydra.core.hydra_config import HydraConfig
        return HydraConfig.get().runtime.output_dir
    except Exception:
        return os.getcwd()


def get_hydra_output_dir(subdir: str | None = None) -> str:
    """
    Возвращает путь к текущей директории вывода Hydra.

    Если указан subdir, возвращает путь к поддиректории внутри вывода,
    создавая её при необходимости.
    """
    base_dir = get_hydra_runtime_output_dir()

    if not subdir:
        return base_dir

    path = os.path.join(base_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path


