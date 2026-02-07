from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.config.types import ExplorationType
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.model.actor import Actor
from nn_laser_stabilizer.rl.policy.deterministic import DeterministicPolicy
from nn_laser_stabilizer.rl.policy.exploration import (
    NoisyExplorationPolicy,
    OrnsteinUhlenbeckExplorationPolicy,
    RandomExplorationPolicy,
)


def make_policy_from_config(
    actor: Actor,
    exploration_config: Config,
) -> Policy:
    exploration_type = ExplorationType.from_str(exploration_config.type) 
    if exploration_type == ExplorationType.NONE:
        return DeterministicPolicy.from_config(
            exploration_config=exploration_config,
            actor=actor,
        )
    elif exploration_type == ExplorationType.RANDOM:
        return RandomExplorationPolicy.from_config(
            exploration_config=exploration_config,
            actor=actor,
        )
    elif exploration_type == ExplorationType.NOISY:
        return NoisyExplorationPolicy.from_config(
            exploration_config=exploration_config,
            actor=actor,
        )
    elif exploration_type == ExplorationType.OU:
        return OrnsteinUhlenbeckExplorationPolicy.from_config(
            exploration_config=exploration_config,
            actor=actor,
        )
    else:
        raise ValueError(f"Unknown exploration type: {exploration_type}")