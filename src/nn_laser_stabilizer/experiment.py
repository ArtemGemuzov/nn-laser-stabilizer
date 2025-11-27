from typing import Callable
from pathlib import Path
from functools import wraps

from nn_laser_stabilizer.config import load_config
from nn_laser_stabilizer.context import ExperimentContext


CONFIGS_DIR = Path("configs")


def experiment(
    relative_config_path: str,
):
    """
    Путь должен относительно configs/
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            absolute_config_path = CONFIGS_DIR / relative_config_path
            
            config = load_config(absolute_config_path, configs_dir=CONFIGS_DIR)
            with ExperimentContext(config) as context:
                return func(context, *args, **kwargs)
        
        return wrapper
    
    return decorator

