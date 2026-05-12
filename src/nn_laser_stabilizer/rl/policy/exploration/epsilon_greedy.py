from typing import Any

import torch

from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy


class EpsilonGreedyPolicy(Policy):
    def __init__(
        self,
        inner: Policy,
        num_actions: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
    ):
        self._inner = inner
        self._num_actions = num_actions
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_steps = epsilon_decay_steps
        self._step = 0
        self._training = True

    @property
    def epsilon(self) -> float:
        if not self._training:
            return 0.0
        t = min(self._step / max(self._epsilon_decay_steps, 1), 1.0)
        return self._epsilon_start + (self._epsilon_end - self._epsilon_start) * t

    def act(self, observation: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        if self._training:
            self._step += 1

        if self._training and torch.rand(1).item() < self.epsilon:
            action = torch.randint(0, self._num_actions, (1,)).float()
            options["policy_info"] = {
                "type": self.__class__.__name__,
                "policy_mode": "train",
                "exploration_applied": True,
                "epsilon": self.epsilon,
                "action": action.tolist(),
            }
            return action, options

        action, options = self._inner.act(observation, options)
        policy_info = dict(options.get("policy_info", {}))
        policy_info["exploration_applied"] = False
        policy_info["epsilon"] = self.epsilon
        options["policy_info"] = policy_info
        return action, options

    def clone(self) -> "EpsilonGreedyPolicy":
        return EpsilonGreedyPolicy(
            inner=self._inner.clone(),
            num_actions=self._num_actions,
            epsilon_start=self._epsilon_start,
            epsilon_end=self._epsilon_end,
            epsilon_decay_steps=self._epsilon_decay_steps,
        )

    def share_memory(self) -> "EpsilonGreedyPolicy":
        self._inner.share_memory()
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._inner.state_dict()

    def load_state_dict(self, state_dict):
        return self._inner.load_state_dict(state_dict)

    def train(self, mode: bool = True) -> "EpsilonGreedyPolicy":
        self._training = mode
        self._inner.train(mode)
        return self

    def eval(self) -> "EpsilonGreedyPolicy":
        return self.train(False)

    def warmup(self, observation_space: Box, num_steps: int = 100) -> None:
        self._inner.warmup(observation_space, num_steps)
