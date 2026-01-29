from pathlib import Path
import argparse
import time

from nn_laser_stabilizer.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.sampler import make_sampler_from_config
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.experiment.workdir_context import WorkingDirectoryContext
from nn_laser_stabilizer.logger import SyncFileLogger, PrefixedLogger
from nn_laser_stabilizer.actor import make_actor_from_config
from nn_laser_stabilizer.critic import make_critic_from_config
from nn_laser_stabilizer.env_wrapper import make_spaces_from_config
from nn_laser_stabilizer.optimizer import Optimizer
from nn_laser_stabilizer.config.config import load_config, find_config_path
from nn_laser_stabilizer.updater import make_updater_from_config


def offline_train(
    config_path: Path,
    buffer_path: Path,
) -> None:
    config_path = find_config_path(config_path)
    config = load_config(config_path)

    buffer_path = buffer_path.resolve()

    with ExperimentContext(config) as context, WorkingDirectoryContext(context.experiment_dir):
        TRAIN_LOG_PREFIX = "TRAIN"
        
        context.logger.log(f"Loading replay buffer from: {buffer_path}")
        buffer = ReplayBuffer.load(buffer_path)
        context.logger.log(f"Replay buffer loaded. Size: {len(buffer)} / capacity={buffer.capacity}")

        observation_space, action_space = make_spaces_from_config(
            context.config.env,
            seed=context.seed,
        )

        network_config = context.config.network

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

        updater_cfg = context.config.updater
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
            sampler_config=context.config.sampler,
        )

        train_log_dir = Path(context.config.training.log_dir)
        train_logger = PrefixedLogger(
            logger=SyncFileLogger(
                log_dir=train_log_dir,
                log_file=context.config.training.log_file,
            ),
            prefix=TRAIN_LOG_PREFIX
        )

        try:
            num_steps = context.config.training.num_steps
            infinite_steps = num_steps == -1

            log_frequency = context.config.training.log_frequency
            logging_enabled = log_frequency > 0

            context.logger.log(
                f"Offline training started "
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

            context.logger.log("Offline training completed.")
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

            train_logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline TD3 training on a pre-collected replay buffer."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pid_delta_tuning",
        help="Relative path to config inside 'configs/' (without .yaml). Default: pid_delta_tuning",
    )
    parser.add_argument(
        "--buffer-path",
        type=str,
        required=True,
        help="Path to saved ReplayBuffer (.pth) file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    offline_train(
        config_path=Path(args.config),
        buffer_path=Path(args.buffer_path),
    )


