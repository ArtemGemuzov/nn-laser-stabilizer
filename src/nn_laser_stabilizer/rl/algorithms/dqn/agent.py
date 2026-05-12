from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.base import Agent
from nn_laser_stabilizer.rl.algorithms.optimizer import Optimizer, SoftUpdater
from nn_laser_stabilizer.rl.algorithms.utils import build_soft_update_pairs
from nn_laser_stabilizer.rl.model.q_network import DiscreteQNetwork
from nn_laser_stabilizer.rl.networks.factory import make_actor_network_from_config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.envs.spaces.discrete import Discrete
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.greedy import GreedyPolicy
from nn_laser_stabilizer.rl.policy.factory import make_exploration_policy_from_config


class DQNAgent(Agent):
    DIR_NAME = "agent_dqn"

    def __init__(
        self,
        q_online: DiscreteQNetwork,
        q_target: DiscreteQNetwork,
        action_space: Discrete,
        optimizer: Optimizer,
        soft_updater: SoftUpdater,
        gamma: float,
    ):
        self._q_online = q_online
        self._q_target = q_target
        self._action_space = action_space
        self._optimizer = optimizer
        self._soft_updater = soft_updater
        self._gamma = gamma

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        observation_space: Box,
        action_space: Discrete,
    ) -> "DQNAgent":
        network_config = algorithm_config.network

        q_network = make_actor_network_from_config(
            network_config=network_config,
            obs_dim=observation_space.dim,
            output_dim=action_space.n,
        )
        q_online = DiscreteQNetwork(network=q_network, num_actions=action_space.n).train()
        q_target = q_online.clone().requires_grad_(False)

        gamma = float(algorithm_config.gamma)
        tau = float(algorithm_config.tau)

        if gamma <= 0.0:
            raise ValueError("algorithm.gamma must be > 0")
        if tau <= 0.0:
            raise ValueError("algorithm.tau must be > 0")

        optimizer = Optimizer(q_online.parameters(), lr=float(algorithm_config.optimizer.lr))
        soft_updater = SoftUpdater(
            pairs=build_soft_update_pairs(module_pairs=((q_target, q_online),)),
            tau=tau,
        )

        return cls(
            q_online=q_online,
            q_target=q_target,
            action_space=action_space,
            optimizer=optimizer,
            soft_updater=soft_updater,
            gamma=gamma,
        )

    def exploration_policy(self, exploration_config: Config) -> Policy:
        base_policy = GreedyPolicy(q_network=self._q_online).train()
        return make_exploration_policy_from_config(
            policy=base_policy,
            action_space=self._action_space,
            exploration_config=exploration_config,
        )

    def default_policy(self) -> Policy:
        return GreedyPolicy(q_network=self._q_online).eval()

    def _q_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_obs: Tensor,
        dones: Tensor,
    ) -> Tensor:
        action_indices = actions.long()

        with torch.no_grad():
            # Double DQN: select with online, evaluate with target
            best_next_actions = self._q_online(next_obs).q_values.argmax(dim=1, keepdim=True)
            next_q = self._q_target(next_obs).q_values.gather(1, best_next_actions)
            target_q = rewards + self._gamma * next_q * (1.0 - dones.float())

        current_q = self._q_online(obs).q_values.gather(1, action_indices)
        return F.mse_loss(current_q, target_q)

    def update_step(self, batch: tuple[Tensor, ...]) -> dict[str, float]:
        obs, actions, rewards, next_obs, dones = batch

        loss = self._q_loss(obs, actions, rewards, next_obs, dones)
        self._optimizer.step(loss)
        self._soft_updater.update()

        return {"loss_q": loss.item()}

    def save(self, path: Path | None = None) -> None:
        path = Path(path) if path is not None else self.default_path
        path.mkdir(parents=True, exist_ok=True)
        self._q_online.save(path / "q_online.pt")
        self._q_target.save(path / "q_target.pt")
        self._optimizer.save(path / "optimizer.pt")

    def load(self, path: Path) -> None:
        path = Path(path)
        self._q_online.load(path / "q_online.pt")
        self._q_target.load(path / "q_target.pt")
        self._optimizer.load(path / "optimizer.pt")
