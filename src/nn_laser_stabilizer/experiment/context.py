from pathlib import Path
from datetime import datetime
import traceback

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.experiment.seed import set_seeds, generate_random_seed
from nn_laser_stabilizer.logger import ConsoleLogger
from nn_laser_stabilizer.paths import EXPERIMENTS_DIR, RESOURCES_DIR


class ExperimentContext:
    def __init__(
        self,
        config: Config,
    ):
        self.config: Config = config

        self._seed: int = 0

    def __enter__(self):
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")

        self._experiment_name = self.config.get("experiment_name", "unnamed_experiment")
        self._experiment_dir: Path = EXPERIMENTS_DIR / self._experiment_name / timestamp
        self._experiment_dir.mkdir(parents=True, exist_ok=True)

        variables = {
            "EXPERIMENT_DIR": str(self._experiment_dir),
            "RESOURCES_DIR": str(RESOURCES_DIR.resolve()),
        }
        self.config = self.config.substitute_placeholders(variables)

        self._set_seed()
        self._save_config()
        self._setup_logger()

        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        self.logger.log(
            f"Experiment started: {self._experiment_name} | "
            f"Start time: {start_time_str} | Directory: {self._experiment_dir}"
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

        if exc_type is not None:
            if exc_type is KeyboardInterrupt:
                self.logger.log("Experiment interrupted by user")
            else:
                formatted_tb = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
                self.logger.log(
                    "Unhandled exception in experiment:\n"
                    f"{formatted_tb}"
                )

        self.logger.log(
            f"Experiment finished: {self._experiment_name} | End time: {end_time_str}"
        )
        self.logger.close()
        
        return True

    def _save_config(self) -> None:
        config_path = self._experiment_dir / "config.yaml"
        self.config.save(config_path)

    def _set_seed(self) -> None:
        self._seed = self.config.get("seed")
        if self._seed is None:
            self._seed = generate_random_seed()
            config_dict = self.config.to_dict()
            config_dict["seed"] = self._seed
            self.config = Config(config_dict)
        set_seeds(self._seed)

    def _setup_logger(self) -> None:
        self._logger = ConsoleLogger(
            log_dir=self._experiment_dir,
            log_file="console.log"
        )

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def logger(self) -> ConsoleLogger:
        return self._logger
    
    @property
    def experiment_dir(self) -> Path:
        return self._experiment_dir


