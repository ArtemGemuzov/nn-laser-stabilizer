from pathlib import Path

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.base import Agent
from nn_laser_stabilizer.rl.model.deterministic_actor import DeterministicActor
from nn_laser_stabilizer.rl.networks.factory import make_actor_network_from_config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.deterministic import DeterministicPolicy
from nn_laser_stabilizer.rl.policy.factory import make_exploration_policy_from_config


class BCAgent(Agent):
    def __init__(self, actor: DeterministicActor, action_space: Box):
        self.actor = actor
        self.action_space = action_space

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        observation_space: Box,
        action_space: Box,
    ) -> "BCAgent":
        actor_config = algorithm_config.actor

        actor_network = make_actor_network_from_config(
            network_config=actor_config.network,
            obs_dim=observation_space.dim,
            output_dim=action_space.dim,
        )
        actor = DeterministicActor(network=actor_network, action_space=action_space).train()

        return cls(actor=actor, action_space=action_space)

    def exploration_policy(self, exploration_config: Config) -> Policy:
        base_policy = DeterministicPolicy(actor=self.actor)
        return make_exploration_policy_from_config(
            policy=base_policy,
            action_space=self.action_space,
            exploration_config=exploration_config,
        )

    def default_policy(self) -> Policy:
        return DeterministicPolicy(actor=self.actor).eval()

    def save_models(self, models_dir: Path) -> None:
        models_dir.mkdir(parents=True, exist_ok=True)
        self.actor.save(models_dir / 'actor.pt')
