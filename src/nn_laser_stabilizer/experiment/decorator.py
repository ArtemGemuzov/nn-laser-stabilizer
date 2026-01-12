from typing import Callable
from functools import wraps
import argparse

from nn_laser_stabilizer.experiment.config import load_config, find_config_path
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.experiment.workdir_context import WorkingDirectoryContext


def experiment(
    relative_config_path: str,
):
    """
    Путь должен относительно configs/
    Может быть переопределен через аргумент командной строки --config
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument(
                "--config",
                type=str,
                default=None,
                help="Relative path to config inside 'configs/' (without .yaml). Overrides default config.",
            )
            parsed_args, _ = parser.parse_known_args()
            
            config_name = parsed_args.config if parsed_args.config is not None else relative_config_path
            absolute_config_path = find_config_path(config_name)
            config = load_config(absolute_config_path)
            
            with ExperimentContext(config) as exp_context, WorkingDirectoryContext(exp_context.experiment_dir):
                return func(exp_context, *args, **kwargs)
        return wrapper
    return decorator

