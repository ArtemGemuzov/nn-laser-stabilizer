import torch

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.exploration.base_exploaration import BaseExplorationPolicy


class NoisyExplorationPolicy(BaseExplorationPolicy):
    def __init__(
        self,
        inner: Policy,
        action_space: Box,
        exploration_steps: int,
        policy_noise: float,
        noise_clip: float,
    ):
        super().__init__(inner, action_space, exploration_steps)
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip

    def _explore(self, action: torch.Tensor, options: dict) -> torch.Tensor:
        noise = (torch.randn_like(action) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
        return self._action_space.clip(action + noise)

    def clone(self) -> "NoisyExplorationPolicy":
        return NoisyExplorationPolicy(
            inner=self._inner.clone(),
            action_space=self._action_space,
            exploration_steps=self._exploration_steps,
            policy_noise=self._policy_noise,
            noise_clip=self._noise_clip,
        )

    @classmethod
    def from_config(cls, exploration_config: Config, *, policy: Policy, action_space: Box) -> "NoisyExplorationPolicy":
        steps = int(exploration_config.steps)
        policy_noise = float(exploration_config.policy_noise)
        noise_clip = float(exploration_config.noise_clip)

        if steps < 0:
            raise ValueError("exploration.steps must be >= 0 for noisy exploration")
        if policy_noise <= 0.0:
            raise ValueError("exploration.policy_noise must be > 0")
        if noise_clip <= 0.0:
            raise ValueError("exploration.noise_clip must be > 0")

        return cls(
            inner=policy,
            action_space=action_space,
            exploration_steps=steps,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
        )
