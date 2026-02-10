from typing import Any

import torch
import torch.nn as nn

from nn_laser_stabilizer.rl.envs.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def tanh_squash(raw_action: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Apply tanh and scale to [low, high]."""
    t = torch.tanh(raw_action)
    return low + (t + 1.0) * (high - low) / 2.0


def gaussian_log_prob(
    normal: torch.distributions.Normal,
    raw_action: torch.Tensor,
) -> torch.Tensor:
    """Log-probability with tanh squashing correction."""
    log_prob = normal.log_prob(raw_action)
    # Enforcing Action Bound: correction for tanh squashing
    log_prob = log_prob - torch.log(1.0 - torch.tanh(raw_action).pow(2) + 1e-6)
    return log_prob.sum(dim=-1, keepdim=True)


class GaussianPolicy(Policy):
    """Stochastic Gaussian policy for SAC.

    Wraps a raw MLP network with ``2 * action_dim`` outputs.
    First ``action_dim`` outputs are interpreted as *mean*,
    the remaining ``action_dim`` as *log_std*.
    """

    def __init__(self, net: nn.Module, action_space: Box):
        self._net = net
        self._action_space = action_space
        self._training = True

    # ------------------------------------------------------------------
    # Public helpers used by SACAgent.forward_train
    # ------------------------------------------------------------------

    def sample(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and return ``(action, log_prob)``."""
        raw = self._net(observation)
        mean, log_std = raw.chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = tanh_squash(x_t, self._action_space.low, self._action_space.high)
        log_prob = gaussian_log_prob(normal, x_t)
        return action, log_prob

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the mean action (no sampling)."""
        raw = self._net(observation)
        mean, _log_std = raw.chunk(2, dim=-1)
        return tanh_squash(mean, self._action_space.low, self._action_space.high)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def act(
        self, observation: torch.Tensor, options: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self._training:
            action, log_prob = self.sample(observation)
            options = {**options, "log_prob": log_prob}
            return action, options
        else:
            action = self.deterministic_action(observation)
            return action, options

    def clone(self) -> "GaussianPolicy":
        import copy

        cloned_net = copy.deepcopy(self._net)
        return GaussianPolicy(net=cloned_net, action_space=self._action_space)

    def share_memory(self) -> "GaussianPolicy":
        self._net.share_memory()
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._net.state_dict()

    def load_state_dict(self, state_dict):
        return self._net.load_state_dict(state_dict)

    def train(self, mode: bool = True) -> "GaussianPolicy":
        self._training = mode
        self._net.train(mode)
        return self

    def eval(self) -> "GaussianPolicy":
        return self.train(False)

    def warmup(self, observation_space: Box, num_steps: int = 100) -> None:
        self._net.eval()
        with torch.no_grad():
            for _ in range(num_steps):
                fake_obs = observation_space.sample()
                self._net(fake_obs)
