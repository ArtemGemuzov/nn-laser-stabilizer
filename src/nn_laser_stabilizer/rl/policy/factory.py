from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.utils.enum import BaseEnum
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.exploration.noisy import NoisyExplorationPolicy
from nn_laser_stabilizer.rl.policy.exploration.ornstein_uhlenbeck import OrnsteinUhlenbeckExplorationPolicy
from nn_laser_stabilizer.rl.policy.exploration.pid import PIDExplorationPolicy
from nn_laser_stabilizer.rl.policy.exploration.random import RandomExplorationPolicy


class ExplorationType(BaseEnum):
    NONE = "none"
    RANDOM = "random"
    NOISY = "noisy"
    OU = "ou"
    PID = "pid"
    SEQUENCE = "sequence"
    

def make_exploration_policy_from_config(
    policy: Policy,
    action_space: Box,
    exploration_config: Config,
) -> Policy:
    exploration_type = ExplorationType.from_str(exploration_config.type)

    if exploration_type == ExplorationType.NONE:
        return policy

    elif exploration_type == ExplorationType.SEQUENCE:
        wrappers = exploration_config.get("wrappers", None)
        if wrappers is None:
            raise ValueError("exploration.wrappers must be provided for exploration.type=sequence")
        if not isinstance(wrappers, list):
            raise ValueError("exploration.wrappers must be a list for exploration.type=sequence")
        wrapped = policy
        for wrapper_cfg in wrappers:
            if not isinstance(wrapper_cfg, dict):
                raise ValueError("Each item in exploration.wrappers must be a dict")
            wrapped = make_exploration_policy_from_config(
                policy=wrapped,
                action_space=action_space,
                exploration_config=Config(wrapper_cfg),
            )
        return wrapped

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
