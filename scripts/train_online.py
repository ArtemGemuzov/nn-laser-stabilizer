from typing import Callable, Dict
from functools import partial
from pathlib import Path
import json
import time

import numpy as np

from nn_laser_stabilizer.experiment.decorator import experiment
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.utils.logger import SyncFileLogger
from nn_laser_stabilizer.rl.data.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.rl.data.sampler import make_sampler_from_config
from nn_laser_stabilizer.rl.collector.collector import make_collector_from_config
from nn_laser_stabilizer.rl.envs.env_wrapper import TorchEnvWrapper, get_spaces_from_config, make_env_from_config
from nn_laser_stabilizer.rl.algorithms.factory import build_algorithm
from nn_laser_stabilizer.rl.policy.policy import Policy


def evaluate(
    policy: Policy,
    env_factory: Callable[[], TorchEnvWrapper],
    num_steps: int,
) -> Dict[str, float]:
    policy.eval()
    env = env_factory()
    
    rewards = np.empty(num_steps)

    obs, _ = env.reset()
    options = {}  
    for i in range(num_steps):
        action, options = policy.act(obs, options) 
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards[i] = reward
        
        if done:
            options = {}
            obs, _ = env.reset()
    
    env.close()
    policy.train()
    return {
        "episodes": rewards.size,
        "reward_mean": rewards.mean(),
        "reward_sum": rewards.sum(),
        "reward_max": rewards.max(),
        "reward_min": rewards.min()
    }


@experiment(
    experiment_name="neural_controller-v1", 
    config_name="neural_controller"
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

    agent, learner = build_algorithm(
        algorithm_config=context.config.algorithm,
        observation_space=observation_space,
        action_space=action_space,
    )

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

    env_config = context.config.env
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
            while infinite_steps or step < num_steps:
                step += 1

                collector.collect(collect_steps_per_iteration)
                
                batch = sampler.sample()

                metrics = learner.update_step(batch)
                
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
                
                if evaluation_enabled and step % evaluation_frequency == 0:
                    eval_metrics = evaluate(
                        agent.default_policy().eval(),
                        lambda: make_env_from_config(env_config),
                        num_steps=evaluation_num_steps,
                    )
                    train_logger.log(json.dumps({
                        "source": LOG_SOURCE,
                        "event": "evaluation",
                        "step": step,
                        "time": time.time(),
                        **eval_metrics,
                    }))
            
            context.logger.log("Training completed.")
            context.logger.log(f"Final buffer size: {len(buffer)}")
        finally:
            context.logger.log("Saving models...")
            models_dir = Path("models")
            agent.save_models(models_dir)
            context.logger.log(f"Models saved to {models_dir}")
            
            context.logger.log("Saving replay buffer...")
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            buffer.save(data_dir / "replay_buffer.pth")
            context.logger.log(f"Replay buffer saved to {data_dir}")
    
    train_logger.close()
    context.logger.log("Collector stopped.")


if __name__ == "__main__":
    main()
