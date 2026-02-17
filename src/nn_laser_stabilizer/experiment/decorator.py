from typing import Callable
from functools import wraps
import argparse

from nn_laser_stabilizer.config.config import Config, find_and_load_config
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.utils.paths import WorkingDirectoryContext


def experiment(
    experiment_name: str | None = None,
    config_name: str | None = None,
    extra_parser: argparse.ArgumentParser | None = None,
):
    """
    Декоратор для запуска функции в контексте эксперимента.

    experiment_name: имя эксперимента (директория в experiments/).
        Задаётся в декораторе или скрипте; при указании переопределяет значение из конфига.
    config_name: имя конфига (строка) — относительно configs/ (без .yaml).
        Может быть переопределён через --config. Если None и --config не передан,
        конфиг собирается из аргументов CLI (extra_parser) и experiment_name.
    extra_parser: опциональный парсер с дополнительными аргументами.
        Распарсенные значения записываются в config.cli (доступны как context.config.cli.<arg>).
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if extra_parser is not None:
                # TODO: надо бы убирать помощь на уровне скриптов
                parser = argparse.ArgumentParser(parents=[extra_parser], add_help=False)
            else:
                parser = argparse.ArgumentParser()
            parser.add_argument(
                "--experiment-name",
                type=str,
                default=None,
                help="Experiment name (subdir in experiments/). Overrides default from decorator/config.",
            )
            parser.add_argument(
                "--config",
                type=str,
                default=None,
                help="Relative path to config inside 'configs/' (without .yaml). Overrides default config.",
            )
            parsed = parser.parse_args()

            resolved_config_name = parsed.config if parsed.config is not None else config_name
            resolved_experiment_name = (
                parsed.experiment_name if parsed.experiment_name is not None 
                else experiment_name if experiment_name is not None 
                else "unnamed_experiment"
            )
            extra_dict = {
                k: v for k, v in vars(parsed).items()
                if k not in ("config", "experiment_name")
            }

            if resolved_config_name is not None:
                config = find_and_load_config(resolved_config_name)
            else:
                config = Config()
            config = config.with_key("cli", extra_dict)
            config = config.with_key("experiment_name", resolved_experiment_name)

            with ExperimentContext(config) as exp_context, WorkingDirectoryContext(exp_context.experiment_dir):
                return func(exp_context, *args, **kwargs)
        return wrapper
    return decorator

