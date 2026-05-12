from typing import Any

import torch

from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.model.q_network import DiscreteQNetwork
from nn_laser_stabilizer.rl.policy.policy import Policy


class GreedyPolicy(Policy):
    def __init__(self, q_network: DiscreteQNetwork):
        self._q_network = q_network
        self._training = False

    @torch.no_grad()
    def act(self, observation: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        state = options.get("hidden_state")
        output = self._q_network(observation, state)
        if output.state is not None:
            options["hidden_state"] = output.state
        action = output.q_values.argmax(dim=-1, keepdim=True).float()
        options["policy_info"] = {
            "type": self.__class__.__name__,
            "policy_mode": "train" if self._training else "eval",
            "action": action.detach().cpu().tolist(),
        }
        return action, options

    def clone(self) -> "GreedyPolicy":
        return GreedyPolicy(q_network=self._q_network.clone())

    def share_memory(self) -> "GreedyPolicy":
        self._q_network.share_memory()
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._q_network.state_dict()

    def load_state_dict(self, state_dict):
        return self._q_network.load_state_dict(state_dict)

    def train(self, mode: bool = True) -> "GreedyPolicy":
        self._training = mode
        self._q_network.train(mode)
        return self

    def eval(self) -> "GreedyPolicy":
        return self.train(False)

    def warmup(self, observation_space: Box, num_steps: int = 100) -> None:
        self._q_network.eval()
        with torch.no_grad():
            for _ in range(num_steps):
                fake_obs = observation_space.sample()
                self._q_network(fake_obs)
