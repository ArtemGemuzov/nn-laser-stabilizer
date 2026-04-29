import argparse
from functools import partial
from pathlib import Path
import json
import time
from collections import deque

from nn_laser_stabilizer.experiment.decorator import experiment
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.utils.logger import SyncFileLogger
from nn_laser_stabilizer.rl.data.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.rl.data.sampler import make_sampler_from_config
from nn_laser_stabilizer.rl.collector.collector import make_collector_from_config
from nn_laser_stabilizer.rl.envs.factory import get_spaces_from_config, make_env_from_config
from nn_laser_stabilizer.rl.algorithms.factory import build_agent


def _format_seconds_human(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes} мин {secs} сек"


def _make_extra_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online training.")
    parser.add_argument(
        "--resume-agent",
        type=Path,
        default=None,
        help="Path to a saved agent directory to resume training from.",
    )
    return parser


@experiment(
    experiment_name="neural_controller-v3", 
    config_name="neural_controller",
    extra_parser=_make_extra_parser(),
)
def main(context: ExperimentContext):
    LOG_SOURCE = "train"

    context.logger.log("Creating components...")

    train_log_dir = Path(context.config.training.log_dir)
    train_logger = SyncFileLogger(log_dir=train_log_dir, log_file=context.config.training.log_file)
    
    observation_space, action_space = get_spaces_from_config(context.config.env, seed=context.seed)
    observation_dim = observation_space.dim
    action_dim = action_space.dim
    
    buffer = ReplayBuffer(
        capacity=context.config.buffer.capacity,
        obs_dim=observation_dim,
        action_dim=action_dim,
    )    

    sampler = make_sampler_from_config(
        buffer=buffer,
        sampler_config=context.config.sampler
    )

    agent = build_agent(
        algorithm_config=context.config.algorithm,
        observation_space=observation_space,
        action_space=action_space,
    )
    resume_agent_path = context.config.cli.get('resume_agent', None)
    if resume_agent_path is not None:
        resume_agent_path = Path(resume_agent_path).resolve()
        if not resume_agent_path.exists():
            raise FileNotFoundError(f"Resume agent path not found: {resume_agent_path}")
        context.logger.log(f"Loading agent from {resume_agent_path}...")
        agent.load(resume_agent_path)
        context.logger.log("Agent loaded. Resuming training.")

    policy = agent.exploration_policy(
        exploration_config=context.config.exploration,
    ).train()
    
    env_factory = partial(make_env_from_config, env_config=context.config.env, seed=context.seed)
    
    collector = make_collector_from_config(
        collector_config=context.config.collector,
        env_factory=env_factory,
        buffer=buffer,
        policy=policy,
        seed=context.seed,
    )

    num_steps = context.config.training.num_steps
    infinite_steps = num_steps == -1
    
    log_frequency = context.config.training.log_frequency
    logging_enabled = log_frequency > 0

    evaluation_frequency = context.config.evaluation.frequency
    evaluation_num_steps = context.config.evaluation.num_steps
    evaluation_enabled = evaluation_num_steps > 0 and evaluation_frequency > 0
    countdown_log_frequency = int(
        context.config.evaluation.get("countdown_log_frequency", log_frequency if log_frequency > 0 else 1)
    )
    countdown_log_enabled = countdown_log_frequency > 0

    final_eval_enabled = bool(context.config.evaluation.get("final.enabled", False))
    final_eval_num_steps = int(
        context.config.evaluation.get("final.num_steps", evaluation_num_steps)
    )

    collect_steps_per_iteration = context.config.collector.collect_steps_per_iteration
    sync_frequency = context.config.collector.sync_frequency

    train_start_step = context.config.training.train_start_step
    sync_start_step = context.config.training.sync_start_step
    
    with collector:
        try:
            context.logger.log("Collector started. Initial data collection...")
            
            collector.ensure(train_start_step)
            
            context.logger.log(f"Initial data collection completed. Buffer size: {len(buffer)}")
            context.logger.log("Training started")

            step = 0
            step_durations = deque(maxlen=20)
            previous_step_ts = time.time()
            while infinite_steps or step < num_steps:
                step += 1

                collector.collect(collect_steps_per_iteration)
                
                batch = sampler.sample()

                metrics = agent.update_step(batch)
                
                if step >= sync_start_step and step % sync_frequency == 0:
                    collector.sync()
                    
                if logging_enabled and step % log_frequency == 0:
                    train_logger.log(json.dumps({
                        "source": LOG_SOURCE,
                        "event": "step",
                        "step": step,
                        "buffer_size": len(buffer),
                        "time": time.time(),
                        **metrics,
                    }))

                step_end_ts = time.time()
                step_duration = step_end_ts - previous_step_ts
                previous_step_ts = step_end_ts
                step_durations.append(step_duration)

                if evaluation_enabled and countdown_log_enabled and step % countdown_log_frequency == 0:
                    steps_until_evaluation = (evaluation_frequency - (step % evaluation_frequency)) % evaluation_frequency
                    if steps_until_evaluation > 0:
                        step_in_evaluation_cycle = step % evaluation_frequency
                        avg_step_duration = sum(step_durations) / len(step_durations) if step_durations else None
                        eta_seconds = (
                            avg_step_duration * steps_until_evaluation
                            if avg_step_duration is not None
                            else None
                        )
                        eta_message = _format_seconds_human(eta_seconds) if eta_seconds is not None else "n/a"
                        context.logger.log(
                            (
                                "Обратный отсчет до оценивания: "
                                f"текущий шаг обучения={step}, "
                                f"шаг в текущем интервале оценивания={step_in_evaluation_cycle}/{evaluation_frequency}, "
                                f"осталось шагов обучения={steps_until_evaluation}, "
                                f"примерное время до начала оценивания={eta_message}"
                            )
                        )
                
                if evaluation_enabled and step % evaluation_frequency == 0:
                    context.logger.log("Оценивание началось")
                    evaluation_start_ts = time.time()
                    eval_metrics = collector.evaluate(num_steps=evaluation_num_steps)
                    evaluation_duration = time.time() - evaluation_start_ts
                    context.logger.log(
                        (
                            "Оценивание завершено: "
                            f"длительность={_format_seconds_human(evaluation_duration)}."
                        )
                    )
                    # Skip evaluation wall-clock time in ETA statistics.
                    previous_step_ts = time.time()
                    train_logger.log(json.dumps({
                        "source": LOG_SOURCE,
                        "event": "evaluation",
                        "step": step,
                        "time": time.time(),
                        **eval_metrics,
                    }))

            context.logger.log("Training completed.")
            
            if final_eval_enabled:
                context.logger.log(
                    f"Running final evaluation ..."
                )
                final_eval_metrics = collector.evaluate(num_steps=final_eval_num_steps)
                train_logger.log(
                    json.dumps(
                        {
                            "source": LOG_SOURCE,
                            "event": "final_evaluation",
                            "step": step,
                            "time": time.time(),
                            **final_eval_metrics,
                        }
                    )
                )

        finally:
            context.logger.log("Saving agent...")
            agent.save()
            context.logger.log(f"Agent saved to {agent.default_path}")
            
            context.logger.log("Saving replay buffer...")
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            buffer.save(data_dir / "replay_buffer.pth")
            context.logger.log(f"Replay buffer saved to {data_dir}")
    
    train_logger.close()
    context.logger.log("Collector stopped.")


if __name__ == "__main__":
    main()
