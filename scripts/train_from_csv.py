from typing import Optional, Tuple
from pathlib import Path
import argparse
import time
from functools import partial

import numpy as np
import pandas as pd

from nn_laser_stabilizer.buffer_loader import load_buffer_from_csv
from nn_laser_stabilizer.sampler import make_sampler_from_config
from nn_laser_stabilizer.loss import make_loss_from_config, TD3BCLoss
from nn_laser_stabilizer.training import td3_train_step, td3bc_train_step
from nn_laser_stabilizer.optimizer import Optimizer, SoftUpdater
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.experiment.workdir_context import WorkingDirectoryContext
from nn_laser_stabilizer.logger import SyncFileLogger, PrefixedLogger
from nn_laser_stabilizer.actor import make_actor_from_config
from nn_laser_stabilizer.critic import make_critic_from_config
from nn_laser_stabilizer.env_wrapper import make_spaces_from_config
from nn_laser_stabilizer.config.config import load_config, find_config_path


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

        loss_module = make_loss_from_config(
            loss_config=base_config.loss,
            actor=actor,
            critic=critic,
            action_space=action_space,
        )

        actor_optimizer = Optimizer(
            loss_module.actor.parameters(),
            lr=base_config.optimizer.actor_lr,
        )
        critic_optimizer = Optimizer(
            list(loss_module.critic1.parameters()) + list(loss_module.critic2.parameters()),
            lr=base_config.optimizer.critic_lr,
        )
        soft_updater = SoftUpdater(loss_module, tau=base_config.optimizer.tau)

        # TODO: временный костыль - выбор функции train step по типу loss. Нужно переделать на правильную абстракцию.
        # Фиксируем loss_module и оптимизаторы с помощью partial
        if isinstance(loss_module, TD3BCLoss):
            train_step = partial(
                td3bc_train_step,
                loss_module=loss_module,
                critic_optimizer=critic_optimizer,
                actor_optimizer=actor_optimizer,
                soft_updater=soft_updater,
            )
        else:
            train_step = partial(
                td3_train_step,
                loss_module=loss_module,
                critic_optimizer=critic_optimizer,
                actor_optimizer=actor_optimizer,
                soft_updater=soft_updater,
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

            policy_freq = base_config.training.policy_freq
            log_frequency = base_config.training.log_frequency
            logging_enabled = log_frequency > 0

            context.logger.log(
                f"Training from CSV started. Buffer size: {len(buffer)}"
            )

            step = 0
            while infinite_steps or step < num_steps:
                step += 1

                batch = sampler.sample()

                update_actor_and_target = (step % policy_freq == 0)

                loss_q1, loss_q2, actor_loss = train_step(
                    batch,
                    update_actor_and_target=update_actor_and_target,
                )

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
            loss_module.actor.save(models_dir / "actor.pth")
            loss_module.critic1.save(models_dir / "critic1.pth")
            loss_module.critic2.save(models_dir / "critic2.pth")
            loss_module.actor_target.save(models_dir / "actor_target.pth")
            loss_module.critic1_target.save(models_dir / "critic1_target.pth")
            loss_module.critic2_target.save(models_dir / "critic2_target.pth")
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
