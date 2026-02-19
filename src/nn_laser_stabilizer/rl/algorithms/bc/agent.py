from pathlib import Path

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.base import Agent
from nn_laser_stabilizer.rl.model.actor import Actor, make_actor_from_config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.deterministic import DeterministicPolicy
from nn_laser_stabilizer.rl.policy.utils import make_policy_from_config


class BCAgent(Agent):
    ACTOR_FILENAME = "actor.pth"

    def __init__(self, actor: Actor):
        self.actor = actor

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        observation_space: Box,
        action_space: Box,
    ) -> "BCAgent":
        actor_config = algorithm_config.actor

        actor = make_actor_from_config(
            network_config=actor_config.network,
            action_space=action_space,
            observation_space=observation_space,
        ).train()

        return cls(actor=actor)

    def exploration_policy(self, exploration_config: Config) -> Policy:
        return make_policy_from_config(
            actor=self.actor,
            exploration_config=exploration_config,
        )

    def default_policy(self) -> Policy:
        return DeterministicPolicy(actor=self.actor).eval()

    def save_models(self, models_dir: Path) -> None:
        models_dir.mkdir(parents=True, exist_ok=True)
        self.actor.save(models_dir / self.ACTOR_FILENAME)
