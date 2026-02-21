from pathlib import Path

import torch.nn.functional as F
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.base import Agent
from nn_laser_stabilizer.rl.algorithms.optimizer import Optimizer
from nn_laser_stabilizer.rl.model.deterministic_actor import DeterministicActor
from nn_laser_stabilizer.rl.networks.factory import make_actor_network_from_config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.deterministic import DeterministicPolicy
from nn_laser_stabilizer.rl.policy.factory import make_exploration_policy_from_config


class BCAgent(Agent):
    DIR_NAME = "agent_bc"

    def __init__(
        self,
        actor: DeterministicActor,
        action_space: Box,
        actor_optimizer: Optimizer,
    ):
        self._actor = actor
        self._action_space = action_space
        self._actor_optimizer = actor_optimizer

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

        actor_optimizer = Optimizer(actor.parameters(), lr=float(actor_config.optimizer.lr))

        return cls(actor=actor, action_space=action_space, actor_optimizer=actor_optimizer)

    def exploration_policy(self, exploration_config: Config) -> Policy:
        base_policy = DeterministicPolicy(actor=self._actor)
        return make_exploration_policy_from_config(
            policy=base_policy,
            action_space=self._action_space,
            exploration_config=exploration_config,
        )

    def default_policy(self) -> Policy:
        return DeterministicPolicy(actor=self._actor).eval()

    def update_step(self, batch: tuple[Tensor, ...]) -> dict[str, float]:
        obs, actions, *_ = batch
        output = self._actor(obs)
        loss = F.mse_loss(output.action, actions)
        self._actor_optimizer.step(loss)
        return {"actor_loss": loss.item()}

    def save(self, path: Path | None = None) -> None:
        path = Path(path) if path is not None else self.default_path
        path.mkdir(parents=True, exist_ok=True)
        self._actor.save(path / 'actor.pt')
        self._actor_optimizer.save(path / 'actor_optimizer.pt')

    def load(self, path: Path) -> None:
        path = Path(path)
        self._actor.load(path / 'actor.pt')
        self._actor_optimizer.load(path / 'actor_optimizer.pt')
