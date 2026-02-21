from functools import partial
from pathlib import Path

from nn_laser_stabilizer.experiment.decorator import experiment
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.rl.algorithms.factory import build_agent
from nn_laser_stabilizer.rl.data.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.rl.collector.collector import make_collector_from_config
from nn_laser_stabilizer.rl.envs.factory import make_env_from_config, get_spaces_from_config


@experiment(
    experiment_name="inference", 
    config_name="inference"
)
def main(context: ExperimentContext):
    context.logger.log("Creating components...")

    env_config = context.config.env

    observation_space, action_space = get_spaces_from_config(env_config, seed=context.seed)

    agent = build_agent(
        algorithm_config=context.config.algorithm,
        observation_space=observation_space,
        action_space=action_space,
    )

    agent_path = Path(context.config.inference.agent_path)
    if not agent_path.exists():
        raise FileNotFoundError(f"Agent path not found: {agent_path}")

    context.logger.log(f"Loading agent from: {agent_path}")
    agent.load(agent_path)
    context.logger.log("Agent loaded successfully")

    policy = agent.default_policy()

    buffer = ReplayBuffer(
        capacity=context.config.buffer.capacity,
        obs_dim=observation_space.dim,
        action_dim=action_space.dim,
    )

    env_factory = partial(make_env_from_config, env_config=env_config, seed=context.seed)

    collector = make_collector_from_config(
        collector_config=context.config.collector,
        env_factory=env_factory,
        buffer=buffer,
        policy=policy,
        seed=context.seed,
    )

    num_steps = context.config.inference.num_steps
    infinite_steps = num_steps == -1

    log_frequency = context.config.inference.log_frequency
    logging_enabled = log_frequency > 0

    collect_steps_per_iteration = context.config.collector.collect_steps_per_iteration

    step = 0

    with collector:
        try:
            context.logger.log("Starting inference...")
            context.logger.log("Press Ctrl+C to stop")

            while infinite_steps or step < num_steps:
                steps_to_collect = collect_steps_per_iteration
                if not infinite_steps:
                    steps_to_collect = min(steps_to_collect, num_steps - step)

                collector.collect(steps_to_collect)
                step += steps_to_collect

                if logging_enabled and step % log_frequency == 0:
                    context.logger.log(
                        f"step={step} buffer_size={len(buffer)}"
                    )

        except KeyboardInterrupt:
            context.logger.log("Interrupted by user")

    context.logger.log("Inference finished")

if __name__ == "__main__":
    main()
