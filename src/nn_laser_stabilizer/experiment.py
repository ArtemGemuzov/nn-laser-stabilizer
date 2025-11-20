from pathlib import Path
from typing import Callable
import yaml
from datetime import datetime
from functools import wraps

from nn_laser_stabilizer.config import Config, load_config
from nn_laser_stabilizer.utils import set_seeds


CONFIGS_DIR = Path("configs")
EXPERIMENTS_DIR = Path("experiments")


class ExperimentContext: 
    def __init__(
        self,
        config: Config,
    ):
        self.config: Config = config
        self.output_base_dir = EXPERIMENTS_DIR
        
        self.experiment_name = config.get("experiment_name", "unnamed_experiment")
        
        self._start_time = datetime.now()
        timestamp = self._start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir : Path = self.output_base_dir / self.experiment_name / timestamp
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        seed = config.get("seed")
        self.seed = seed
        if seed is not None:
            set_seeds(seed)
        
        self.save_config()
    
    def __enter__(self):
        start_time_str = self._start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Experiment started: {self.experiment_name} | Start time: {start_time_str} | Directory: {self.experiment_dir}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

        print(f"Experiment finished: {self.experiment_name} | End time: {end_time_str}")
        
        if exc_type is KeyboardInterrupt:
            return True  
        return False
    
    def save_config(self) -> None:
        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def get_path(self, *subdirs: str) -> Path:
        path = self.experiment_dir
        for subdir in subdirs:
            path = path / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def logs_dir(self) -> Path:
        return self.get_path("logs")
    
    @property
    def models_dir(self) -> Path:
        return self.get_path("models")
    
    @property
    def data_dir(self) -> Path:
        return self.get_path("data")


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
            
            config = load_config(absolute_config_path)
            with ExperimentContext(config) as context:
                return func(context, *args, **kwargs)
        
        return wrapper
    
    return decorator

