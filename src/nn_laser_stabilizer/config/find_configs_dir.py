from pathlib import Path

DEFAULT_CONGIFS_DIR_NAME = "configs"

def find_configs_dir(target: str = DEFAULT_CONGIFS_DIR_NAME) -> str:
    start = Path(__file__).resolve()

    current = start
    while current != current.parent:
        candidate = current / target
        if candidate.is_dir():
            target_path = candidate.resolve()
            return str(target_path)
        current = current.parent

    raise FileNotFoundError(f"Директория '{target}' не найдена при поиске, начиная от {start}")