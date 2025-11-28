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
    source_config_path = Path(context.config.inference.source_config_path)
    if not source_config_path.exists():
        raise FileNotFoundError(f"Source config file not found: {source_config_path}")
    
    source_config = load_config(source_config_path)
    context.logger.log(f"Loading config from: {source_config_path}")
    
    env_config = context.config.env
    env_factory = partial(
        make_env_from_config,
        env_config=env_config,
        seed=context.seed
    )
    env = env_factory()

    actor_path = Path(context.config.inference.actor_path)
    if not actor_path.exists():
        raise FileNotFoundError(f"Actor model file not found: {actor_path}")
    
    context.logger.log(f"Loading actor from: {actor_path}")
    
    actor = load_actor_from_path(actor_path, source_config.network)
    actor.eval()
    
    context.logger.log("Actor loaded successfully")
    
    num_steps = context.config.inference.num_steps
    infinite_steps = num_steps == -1
    
    log_frequency = context.config.inference.log_frequency
    logging_enabled = log_frequency > 0
    
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
        while infinite_steps or step < num_steps:
            action, options = policy.act(obs, options)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            step += 1
            episode_reward += reward.item()
            
            if logging_enabled and step % log_frequency == 0:
                context.logger.log(
                    f"step={step} episode={episode_count} "
                    f"episode_reward={episode_reward:.4f}"
                )
            
            if done:
                episode_count += 1

                if logging_enabled:
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
