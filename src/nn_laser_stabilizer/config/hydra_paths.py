import os


def get_hydra_output_dir(subdir: str | None = None) -> str:
    """
    Возвращает путь к текущей директории вывода Hydra.

    Если указан subdir, возвращает путь к поддиректории внутри вывода,
    создавая её при необходимости.
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        base_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        base_dir = os.getcwd()

    if not subdir:
        return base_dir

    path = os.path.join(base_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path


