from typing import Callable
from functools import wraps

from nn_laser_stabilizer.config import load_config, find_config_path
from nn_laser_stabilizer.context import ExperimentContext


def experiment(
    relative_config_path: str,
):
    """
    Путь должен относительно configs/
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            absolute_config_path = find_config_path(relative_config_path)
            config = load_config(absolute_config_path)
            
            with ExperimentContext(config) as context:
                return func(context, *args, **kwargs)
        return wrapper
    return decorator

