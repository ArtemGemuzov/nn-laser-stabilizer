from pathlib import Path

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.base import Agent
from nn_laser_stabilizer.rl.model.deterministic_actor import DeterministicActor
from nn_laser_stabilizer.rl.model.critic import Critic
from nn_laser_stabilizer.rl.networks.factory import make_actor_network_from_config, make_critic_network_from_config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.deterministic import DeterministicPolicy
from nn_laser_stabilizer.rl.policy.factory import make_exploration_policy_from_config


class TD3Agent(Agent):
    def __init__(self, actor: DeterministicActor, critic: Critic, action_space: Box):
        self.actor = actor
        self.critic1 = critic
        self.critic2 = critic.clone(reinitialize_weights=True)

        self.actor_target = self.actor.clone().requires_grad_(False)
        self.critic1_target = self.critic1.clone().requires_grad_(False)
        self.critic2_target = self.critic2.clone().requires_grad_(False)

        self.action_space = action_space

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        observation_space: Box,
        action_space: Box,
    ) -> "TD3Agent":
        actor_config = algorithm_config.actor
        critic_config = algorithm_config.critic

        actor_network = make_actor_network_from_config(
            network_config=actor_config.network,
            obs_dim=observation_space.dim,
            output_dim=action_space.dim,
        )
        actor = DeterministicActor(network=actor_network, action_space=action_space).train()

        critic_network = make_critic_network_from_config(
            network_config=critic_config.network,
            obs_dim=observation_space.dim,
            action_dim=action_space.dim,
        )
        critic = Critic(network=critic_network).train()

        return cls(actor=actor, critic=critic, action_space=action_space)

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
        self.critic1.save(models_dir / 'critic1.pt')
        self.critic2.save(models_dir / 'critic2.pt')
        self.actor_target.save(models_dir / 'actor_target.pt')
        self.critic1_target.save(models_dir / 'critic1_target.pt')
        self.critic2_target.save(models_dir / 'critic2_target.pt')
