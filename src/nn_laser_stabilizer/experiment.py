from typing import Callable, Optional
from pathlib import Path
import yaml
from datetime import datetime
from functools import wraps

from nn_laser_stabilizer.config import Config, load_config
from nn_laser_stabilizer.seed import set_seeds, generate_random_seed
from nn_laser_stabilizer.logger import ConsoleLogger


CONFIGS_DIR = Path("configs")
EXPERIMENTS_DIR = Path("experiments")


class ExperimentContext: 
    def __init__(
        self,
        config: Config,
    ):
        self.config: Config = config

        self._seed: int = 0
        self._console_logger: Optional[ConsoleLogger] = None
    
    def __enter__(self):
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")

        self._experiment_name = self.config.get("experiment_name", "unnamed_experiment")
        self._experiment_dir : Path = EXPERIMENTS_DIR / self._experiment_name / timestamp
        self._experiment_dir.mkdir(parents=True, exist_ok=True)
        
        variables = {
            "EXPERIMENT_DIR": str(self._experiment_dir),
            "EXPERIMENT_NAME": self._experiment_name,
            "TIMESTAMP": timestamp,
        }
        self.config = self.config.substitute_placeholders(variables)
        
        self._set_seed()
        self._save_config()
        self._setup_logger()
        
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        self.console_logger.log(f"Experiment started: {self._experiment_name} | Start time: {start_time_str} | Directory: {self._experiment_dir}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

        self.console_logger.log(f"Experiment finished: {self._experiment_name} | End time: {end_time_str}")
        self.console_logger.close()
        
        if exc_type is KeyboardInterrupt:
            return True  
        return False
    
    def _save_config(self) -> None:
        config_path = self._experiment_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _set_seed(self) -> None:
        self._seed = self.config.get("seed")
        if self._seed is None:
            self._seed = generate_random_seed()
            config_dict = self.config.to_dict()
            config_dict["seed"] = self._seed
            self.config = Config(config_dict)
        set_seeds(self._seed)
    
    def _setup_logger(self) -> None:
        self._console_logger = ConsoleLogger(
            log_dir=self._experiment_dir,
            log_file="console.log"
        )

    def _get_path(self, *subdirs: str) -> Path:
        path = self._experiment_dir
        for subdir in subdirs:
            path = path / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def experiment_dir(self) -> Path:
        return self._experiment_dir
    
    @property
    def models_dir(self) -> Path:
        return self._get_path("models")
    
    @property
    def data_dir(self) -> Path:
        return self._get_path("data")
    
    @property
    def seed(self) -> int:
        return self._seed
    
    @property
    def console_logger(self) -> ConsoleLogger:
        return self._console_logger


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

