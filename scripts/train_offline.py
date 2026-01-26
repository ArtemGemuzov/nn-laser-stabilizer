from pathlib import Path
import argparse
import time
from functools import partial

from nn_laser_stabilizer.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.sampler import make_sampler_from_config
from nn_laser_stabilizer.loss import make_loss_from_config, TD3Loss, TD3BCLoss
from nn_laser_stabilizer.training import td3_train_step, td3bc_train_step
from nn_laser_stabilizer.optimizer import Optimizer, SoftUpdater
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.experiment.workdir_context import WorkingDirectoryContext
from nn_laser_stabilizer.logger import SyncFileLogger, PrefixedLogger
from nn_laser_stabilizer.actor import make_actor_from_config
from nn_laser_stabilizer.critic import make_critic_from_config
from nn_laser_stabilizer.env_wrapper import make_spaces_from_config
from nn_laser_stabilizer.config.config import load_config, find_config_path


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

        loss_module = make_loss_from_config(
            loss_config=context.config.loss,
            actor=actor,
            critic=critic,
            action_space=action_space,
        )

        actor_optimizer = Optimizer(
            loss_module.actor.parameters(),
            lr=context.config.optimizer.actor_lr,
        )
        critic_optimizer = Optimizer(
            list(loss_module.critic1.parameters()) + list(loss_module.critic2.parameters()),
            lr=context.config.optimizer.critic_lr,
        )
        soft_updater = SoftUpdater(loss_module, tau=context.config.optimizer.tau)

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

            policy_freq = context.config.training.policy_freq
            log_frequency = context.config.training.log_frequency
            logging_enabled = log_frequency > 0

            context.logger.log(
                f"Offline training started "
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

            context.logger.log("Offline training completed.")
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


