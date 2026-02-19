from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.config.types import ExplorationType
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.exploration.noisy import NoisyExplorationPolicy
from nn_laser_stabilizer.rl.policy.exploration.ornstein_uhlenbeck import OrnsteinUhlenbeckExplorationPolicy
from nn_laser_stabilizer.rl.policy.exploration.pid import PIDExplorationPolicy
from nn_laser_stabilizer.rl.policy.exploration.random import RandomExplorationPolicy


def make_exploration_policy_from_config(
    policy: Policy,
    action_space: Box,
    exploration_config: Config,
) -> Policy:
    exploration_type = ExplorationType.from_str(exploration_config.type)

    if exploration_type == ExplorationType.NONE:
        return policy

    elif exploration_type == ExplorationType.RANDOM:
        return RandomExplorationPolicy.from_config(
            exploration_config, policy=policy, action_space=action_space,
        )

    elif exploration_type == ExplorationType.NOISY:
        return NoisyExplorationPolicy.from_config(
            exploration_config, policy=policy, action_space=action_space,
        )

    elif exploration_type == ExplorationType.OU:
        return OrnsteinUhlenbeckExplorationPolicy.from_config(
            exploration_config, policy=policy, action_space=action_space,
        )

    elif exploration_type == ExplorationType.PID:
        return PIDExplorationPolicy.from_config(
            exploration_config, policy=policy, action_space=action_space,
        )

    else:
        raise ValueError(f"Unknown exploration type: {exploration_type}")
