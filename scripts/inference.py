from pathlib import Path

from nn_laser_stabilizer.experiment.decorator import experiment
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.rl.algorithms.factory import build_agent
from nn_laser_stabilizer.rl.envs.factory import make_env_from_config, get_spaces_from_config


@experiment(
    experiment_name="inference", 
    config_name="pid_delta_tuning-inference"
)
def main(context: ExperimentContext):
    env_config = context.config.env

    observation_space, action_space = get_spaces_from_config(env_config, seed=context.seed)

    agent = build_agent(
        algorithm_config=context.config.algorithm,
        observation_space=observation_space,
        action_space=action_space,
    )

    models_dir = Path(context.config.inference.models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    context.logger.log(f"Loading models from: {models_dir}")
    agent.load(models_dir)
    context.logger.log("Models loaded successfully")

    policy = agent.default_policy()

    num_steps = context.config.inference.num_steps
    infinite_steps = num_steps == -1

    log_frequency = context.config.inference.log_frequency
    logging_enabled = log_frequency > 0

    step = 0
    episode_reward = 0.0
    episode_count = 0
    options = {}

    env = make_env_from_config(env_config=env_config, seed=context.seed)

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
