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
        *,
        start_step: int,
        end_step: int | None,
        policy_noise: float,
        noise_clip: float,
    ):
        super().__init__(inner, action_space, start_step=start_step, end_step=end_step)
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip

    def _explore(self, action: torch.Tensor, options: dict) -> torch.Tensor:
        noise = (torch.randn_like(action) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
        return self._action_space.clip(action + noise)

    def clone(self) -> "NoisyExplorationPolicy":
        return NoisyExplorationPolicy(
            inner=self._inner.clone(),
            action_space=self._action_space,
            start_step=self._start_step,
            end_step=self._end_step,
            policy_noise=self._policy_noise,
            noise_clip=self._noise_clip,
        )

    @classmethod
    def from_config(cls, exploration_config: Config, *, policy: Policy, action_space: Box) -> "NoisyExplorationPolicy":
        start_step = int(exploration_config.get("start_step", 0))
        steps = int(exploration_config.get("steps", 0))
        end_step_raw = exploration_config.get("end_step", None)
        end_step = None if end_step_raw is None else int(end_step_raw)
        policy_noise = float(exploration_config.policy_noise)
        noise_clip = float(exploration_config.noise_clip)

        if start_step < 0:
            raise ValueError("exploration.start_step must be >= 0 for noisy exploration")
        if steps < 0:
            raise ValueError("exploration.steps must be >= 0 for noisy exploration")
        if end_step is None:
            end_step = start_step + steps
        if end_step < start_step:
            raise ValueError("exploration.end_step must be >= exploration.start_step for noisy exploration")
        if policy_noise <= 0.0:
            raise ValueError("exploration.policy_noise must be > 0")
        if noise_clip <= 0.0:
            raise ValueError("exploration.noise_clip must be > 0")

        return cls(
            inner=policy,
            action_space=action_space,
            start_step=start_step,
            end_step=end_step,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
        )
