from pathlib import Path

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.base import Agent
from nn_laser_stabilizer.rl.model.actor import Actor, make_actor_from_config
from nn_laser_stabilizer.rl.model.critic import Critic, make_critic_from_config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.deterministic import DeterministicPolicy
from nn_laser_stabilizer.rl.policy.utils import make_policy_from_config


class TD3Agent(Agent):
    ACTOR_FILENAME = "actor.pth"
    CRITIC1_FILENAME = "critic1.pth"
    CRITIC2_FILENAME = "critic2.pth"
    ACTOR_TARGET_FILENAME = "actor_target.pth"
    CRITIC1_TARGET_FILENAME = "critic1_target.pth"
    CRITIC2_TARGET_FILENAME = "critic2_target.pth"

    def __init__(self, actor: Actor, critic: Critic):
        self.actor = actor
        self.critic1 = critic
        self.critic2 = critic.clone(reinitialize_weights=True)

        self.actor_target = self.actor.clone().requires_grad_(False)
        self.critic1_target = self.critic1.clone().requires_grad_(False)
        self.critic2_target = self.critic2.clone().requires_grad_(False)

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        observation_space: Box,
        action_space: Box,
    ) -> "TD3Agent":
        actor_config = algorithm_config.actor
        critic_config = algorithm_config.critic

        actor = make_actor_from_config(
            network_config=actor_config.network,
            action_space=action_space,
            observation_space=observation_space,
        ).train()

        critic = make_critic_from_config(
            network_config=critic_config.network,
            obs_dim=observation_space.dim,
            action_dim=action_space.dim,
        ).train()

        return cls(actor=actor, critic=critic)

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
        self.critic1.save(models_dir / self.CRITIC1_FILENAME)
        self.critic2.save(models_dir / self.CRITIC2_FILENAME)
        self.actor_target.save(models_dir / self.ACTOR_TARGET_FILENAME)
        self.critic1_target.save(models_dir / self.CRITIC1_TARGET_FILENAME)
        self.critic2_target.save(models_dir / self.CRITIC2_TARGET_FILENAME)
