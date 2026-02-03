from pathlib import Path
import argparse
import time

from nn_laser_stabilizer.data.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.data.sampler import make_sampler_from_config
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.paths import WorkingDirectoryContext
from nn_laser_stabilizer.logger import SyncFileLogger, PrefixedLogger
from nn_laser_stabilizer.model.actor import make_actor_from_config
from nn_laser_stabilizer.model.critic import make_critic_from_config
from nn_laser_stabilizer.envs.env_wrapper import make_spaces_from_config
from nn_laser_stabilizer.optimizer import Optimizer
from nn_laser_stabilizer.config.config import load_config, find_config_path
from nn_laser_stabilizer.algorithm.algorithm import make_updater_from_config


def train_from_buffer(config_path: Path, buffer_path: Path) -> None:
    config_path = find_config_path(config_path)
    config = load_config(config_path)

    buffer_path = buffer_path.resolve()
    if not buffer_path.exists():
        raise FileNotFoundError(f"Buffer file not found: {buffer_path}")

    with ExperimentContext(config) as context, WorkingDirectoryContext(context.experiment_dir):
        TRAIN_LOG_PREFIX = "TRAIN"

        context.logger.log(f"Loading buffer: {buffer_path}")
        buffer = ReplayBuffer.load(buffer_path)
        context.logger.log(f"Buffer loaded. Size: {len(buffer)} / capacity={buffer.capacity}")

        observation_space, action_space = make_spaces_from_config(
            config.env,
            seed=context.seed,
        )

        network_config = config.network

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

        updater_cfg = config.updater
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
            sampler_config=config.sampler,
        )

        train_log_dir = Path(config.training.log_dir)
        train_logger = PrefixedLogger(
            logger=SyncFileLogger(
                log_dir=train_log_dir,
                log_file=config.training.log_file,
            ),
            prefix=TRAIN_LOG_PREFIX,
        )

        try:
            num_steps = config.training.num_steps
            infinite_steps = num_steps == -1

            log_frequency = config.training.log_frequency
            logging_enabled = log_frequency > 0

            context.logger.log(f"Training from buffer started. Buffer size: {len(buffer)}")

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

            context.logger.log("Training from buffer completed.")
        finally:
            context.logger.log("Saving models...")
            models_dir = Path("models")
            updater.save_models(models_dir)
            context.logger.log(f"Models saved to {models_dir}")

            buffer_dir = Path("data")
            buffer_dir.mkdir(parents=True, exist_ok=True)
            buffer.save(buffer_dir / "replay_buffer.pth")
            context.logger.log(f"ReplayBuffer saved to {buffer_dir}")

            train_logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline training from replay buffer file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Relative path to *flat* training config inside 'configs/' (without .yaml).",
    )
    parser.add_argument(
        "--buffer",
        type=Path,
        required=True,
        help="Path to .pth replay buffer file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_from_buffer(
        config_path=Path(args.config),
        buffer_path=args.buffer,
    )
