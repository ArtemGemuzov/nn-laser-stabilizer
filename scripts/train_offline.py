import argparse
import time
from pathlib import Path

from nn_laser_stabilizer.experiment.decorator import experiment
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.utils.logger import SyncFileLogger, PrefixedLogger
from nn_laser_stabilizer.rl.data.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.rl.data.sampler import make_sampler_from_config
from nn_laser_stabilizer.rl.envs.factory import get_spaces_from_config
from nn_laser_stabilizer.rl.algorithms.factory import build_agent


def _make_extra_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline TD3 training on a pre-collected replay buffer."
    )
    parser.add_argument(
        "--buffer-path",
        type=Path,
        required=True,
        help="Path to saved ReplayBuffer (.pth) file.",
    )
    return parser


@experiment("train_offline", "pid_delta_tuning", extra_parser=_make_extra_parser())
def main(context: ExperimentContext) -> None:
    config = context._config
    buffer_path = Path(context._config.cli.buffer_path).resolve()

    TRAIN_LOG_PREFIX = "TRAIN"
    context.logger.log(f"Loading replay buffer from: {buffer_path}")
    buffer = ReplayBuffer.load(buffer_path)
    context.logger.log(f"Replay buffer loaded. Size: {len(buffer)} / capacity={buffer.capacity}")

    observation_space, action_space = get_spaces_from_config(config.env, seed=context.seed)

    agent = build_agent(
        algorithm_config=config.algorithm,
        observation_space=observation_space,
        action_space=action_space,
    )

    sampler = make_sampler_from_config(buffer=buffer, sampler_config=config.sampler)

    train_log_dir = Path(config.training.log_dir)
    train_logger = PrefixedLogger(
        logger=SyncFileLogger(log_dir=train_log_dir, log_file=config.training.log_file),
        prefix=TRAIN_LOG_PREFIX,
    )

    try:
        num_steps = config.training.num_steps
        infinite_steps = num_steps == -1
        log_frequency = config.training.log_frequency
        logging_enabled = log_frequency > 0

        context.logger.log("Offline training started.")

        step = 0
        while infinite_steps or step < num_steps:
            step += 1
            batch = sampler.sample()
            metrics = agent.update_step(batch)

            if logging_enabled and step % log_frequency == 0:
                timestamp = time.time()
                metrics_str = " ".join(f"{k}={v}" for k, v in metrics.items())
                train_logger.log(
                    f"step: {metrics_str} buffer_size={len(buffer)} step={step} time={timestamp}"
                )

        context.logger.log("Offline training completed.")
    finally:
        context.logger.log("Saving agent...")
        agent.save()
        context.logger.log(f"Agent saved to {agent.default_path}")

        train_logger.close()


if __name__ == "__main__":
    main()
