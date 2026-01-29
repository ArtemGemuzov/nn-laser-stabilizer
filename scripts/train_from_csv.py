from typing import Optional, Tuple
from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd

from nn_laser_stabilizer.data.buffer_loader import load_buffer_from_csv
from nn_laser_stabilizer.data.sampler import make_sampler_from_config
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.paths import WorkingDirectoryContext
from nn_laser_stabilizer.logger import SyncFileLogger, PrefixedLogger
from nn_laser_stabilizer.model.actor import make_actor_from_config
from nn_laser_stabilizer.model.critic import make_critic_from_config
from nn_laser_stabilizer.envs.env_wrapper import make_spaces_from_config
from nn_laser_stabilizer.optimizer import Optimizer
from nn_laser_stabilizer.config.config import load_config, find_config_path
from nn_laser_stabilizer.algorithm.utils import make_updater_from_config


def make_extract_transition(
    process_variable_max: float,
    control_min: int,
    control_max: int,
    setpoint: int,
):
    span = float(control_max - control_min)
    setpoint_norm = setpoint / process_variable_max

    def _normalize_control_output(control_output: int) -> float:
        norm_01 = float(control_output - control_min) / span
        return 2.0 * norm_01 - 1.0

    def extract_transition(
        df: pd.DataFrame,
        idx: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        if idx == 0 or idx >= len(df) - 1:
            return None

        current_row = df.iloc[idx]
        next_row = df.iloc[idx + 1]
        prev_row = df.iloc[idx - 1]

        process_variable = float(current_row["process_variable"])
        process_variable_norm = np.clip(process_variable / process_variable_max, 0.0, 1.0)
        error = setpoint_norm - process_variable_norm

        control_output_prev = int(prev_row["control_output"])
        control_output_prev_norm = _normalize_control_output(control_output_prev)

        observation = np.array([error, control_output_prev_norm], dtype=np.float32)

        control_output_t = int(current_row["control_output"])
        control_output_t_norm = _normalize_control_output(control_output_t)

        action = np.array([control_output_t_norm], dtype=np.float32)

        next_process_variable = float(next_row["process_variable"])
        next_process_variable_norm = np.clip(next_process_variable / process_variable_max, 0.0, 1.0)
        next_error = setpoint_norm - next_process_variable_norm

        next_observation = np.array([next_error, control_output_t_norm], dtype=np.float32)

        reward = 1.0 - 2.0 * abs(next_error)

        done = False

        return observation, action, float(reward), next_observation, done

    return extract_transition


def train_from_csv(
    config_path: Path
) -> None:
    config_path = find_config_path(config_path)
    config = load_config(config_path)
    
    base_config_path = config.base_config
    base_config = load_config(find_config_path(base_config_path))

    csv_data = config.csv_data
    csv_path = Path(csv_data.csv_path)
    skip_rows = int(csv_data.skip_rows)

    csv_path = csv_path.resolve()

    with ExperimentContext(config) as context, WorkingDirectoryContext(context.experiment_dir):
        TRAIN_LOG_PREFIX = "TRAIN"
        
        context.logger.log(f"Loading CSV file: {csv_path}")
        
        df = pd.read_csv(csv_path)
        context.logger.log(f"CSV file loaded. Total rows: {len(df)}")
        
        if len(df) > skip_rows:
            df = df.iloc[skip_rows:]
            context.logger.log(f"Skipped first {skip_rows} rows. Using {len(df)} rows")
        else:
            context.logger.log(f"Warning: CSV has only {len(df)} rows, cannot skip {skip_rows} rows")
            raise ValueError(f"CSV file has {len(df)} rows, but need to skip {skip_rows} rows")
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_csv_path = Path(tmp_file.name)
        
        try:
            process_variable_max = float(base_config.env.args.process_variable_max)
            control_min = int(base_config.env.args.control_min)
            control_max = int(base_config.env.args.control_max)
            setpoint = int(base_config.env.args.setpoint)
            
            context.logger.log("Converting CSV to ReplayBuffer...")
            extract_transition_fn = make_extract_transition(
                process_variable_max=process_variable_max,
                control_min=control_min,
                control_max=control_max,
                setpoint=setpoint,
            )
            buffer = load_buffer_from_csv(
                csv_path=tmp_csv_path,
                extract_transition=extract_transition_fn,
            )
            context.logger.log(f"ReplayBuffer created. Size: {len(buffer)} / capacity={buffer.capacity}")
        finally:
            tmp_csv_path.unlink()

        observation_space, action_space = make_spaces_from_config(
            base_config.env,
            seed=context.seed,
        )

        network_config = base_config.network

        actor = make_actor_from_config(
            network_config=network_config,
            action_space=action_space,
            observation_space=observation_space,
        ).train()

        critic = make_critic_from_config(
            network_config=network_config,
            obs_dim=observation_space.dim,
            action_dim=action_space.dim,
        ).train()

        updater_cfg = base_config.updater
        updater = make_updater_from_config(
            updater_config=updater_cfg,
            actor=actor,
            critic=critic,
            actor_optimizer_factory=lambda params: Optimizer(
                params,
                lr=updater_cfg.actor_lr,
            ),
            critic_optimizer_factory=lambda params: Optimizer(
                params,
                lr=updater_cfg.critic_lr,
            ),
        )

        sampler = make_sampler_from_config(
            buffer=buffer,
            sampler_config=base_config.sampler,
        )

        train_log_dir = Path(base_config.training.log_dir)
        train_logger = PrefixedLogger(
            logger=SyncFileLogger(
                log_dir=train_log_dir,
                log_file=base_config.training.log_file,
            ),
            prefix=TRAIN_LOG_PREFIX
        )

        try:
            num_steps = base_config.training.num_steps
            infinite_steps = num_steps == -1

            log_frequency = base_config.training.log_frequency
            logging_enabled = log_frequency > 0

            context.logger.log(
                f"Training from CSV started. Buffer size: {len(buffer)}"
            )

            step = 0
            while infinite_steps or step < num_steps:
                step += 1

                batch = sampler.sample()

                loss_q1, loss_q2, actor_loss = updater.update_step(batch)

                if logging_enabled and step % log_frequency == 0:
                    timestamp = time.time()
                    if actor_loss is not None:
                        train_logger.log(
                            f"step: actor_loss={actor_loss} buffer_size={len(buffer)} "
                            f"loss_q1={loss_q1} loss_q2={loss_q2} step={step} time={timestamp}"
                        )
                    else:
                        train_logger.log(
                            f"step: buffer_size={len(buffer)} "
                            f"loss_q1={loss_q1} loss_q2={loss_q2} step={step} time={timestamp}"
                        )

            context.logger.log("Training from CSV completed.")
        finally:
            context.logger.log("Saving models...")
            models_dir = Path("models")
            models_dir.mkdir(parents=True, exist_ok=True)
            updater.actor.save(models_dir / "actor.pth")
            updater.critic1.save(models_dir / "critic1.pth")
            updater.critic2.save(models_dir / "critic2.pth")
            updater.actor_target.save(models_dir / "actor_target.pth")
            updater.critic1_target.save(models_dir / "critic1_target.pth")
            updater.critic2_target.save(models_dir / "critic2_target.pth")
            context.logger.log(f"Models saved to {models_dir}")

            buffer_dir = Path("data")
            buffer.save(buffer_dir / "replay_buffer.pth")
            context.logger.log(f"ReplayBuffer saved to {buffer_dir}")

            train_logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline training from CSV file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="train_from_csv",
        help="Relative path to config inside 'configs/' (without .yaml). Default: train_from_csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_from_csv(
        config_path=Path(args.config),
    )
