from functools import partial
from pathlib import Path

from nn_laser_stabilizer.actor import load_actor_from_path
from nn_laser_stabilizer.policy import DeterministicPolicy
from nn_laser_stabilizer.env_wrapper import make_env_from_config
from nn_laser_stabilizer.experiment import experiment
from nn_laser_stabilizer.context import ExperimentContext
from nn_laser_stabilizer.config import load_config


@experiment("pid_delta_tuning-inference")
def main(context: ExperimentContext):
    source_experiment_dir = Path(context.config.inference.source_experiment_dir)
    if not source_experiment_dir.exists():
        raise FileNotFoundError(f"Source experiment directory not found: {source_experiment_dir}")
    
    source_config = load_config(source_experiment_dir / "config.yaml")
    context.logger.log(f"Loading actor from source experiment: {source_experiment_dir}")
    
    env_config = context.config.env
    env_factory = partial(
        make_env_from_config,
        env_config=env_config,
        seed=context.seed
    )
    env = env_factory()
    
    actor_path = source_experiment_dir / "models" / "actor.pth"
    actor = load_actor_from_path(actor_path, source_config.network)
    actor.eval()
    
    context.logger.log("Actor loaded successfully")
    
    num_steps = context.config.inference.num_steps
    log_frequency = context.config.inference.log_frequency
    
    step = 0
    episode_reward = 0.0
    episode_count = 0

    policy = DeterministicPolicy(actor).eval()
    options = {}
    
    context.logger.log("Starting environment...")
    obs, _ = env.reset()
    context.logger.log("Environment started successfully")
    
    context.logger.log("Starting inference...")
    context.logger.log("Press Ctrl+C to stop")
    
    try:
        while num_steps is None or step < num_steps:
            action, options = policy.act(obs, options)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            step += 1
            episode_reward += reward.item()
            
            if log_frequency is not None and step % log_frequency == 0:
                context.logger.log(
                    f"step={step} episode={episode_count} "
                    f"episode_reward={episode_reward:.4f}"
                )
            
            if done:
                episode_count += 1

                if log_frequency is not None:
                    context.logger.log(
                        f"Episode {episode_count} finished | "
                        f"Steps: {step} | "
                        f"Episode reward: {episode_reward:.4f}"
                    )

                episode_reward = 0.0
                options = {}
                obs, _ = env.reset()
    
    finally:
        context.logger.log("Inference finished")
        context.logger.log(f"Total steps: {step}")
        context.logger.log(f"Total episodes: {episode_count}")
        
        env.close()


if __name__ == "__main__":
    main()
