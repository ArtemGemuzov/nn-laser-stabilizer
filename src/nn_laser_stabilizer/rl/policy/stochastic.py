from typing import Any

import torch

from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.model.stochastic_actor import StochasticActor, StochasticActorOutput
from nn_laser_stabilizer.rl.policy.policy import Policy


class StochasticPolicy(Policy):
    def __init__(self, actor: StochasticActor):
        self._actor = actor
        self._training = True

    @torch.no_grad()
    def act(self, observation: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        state = options.get('hidden_state')
        output : StochasticActorOutput = self._actor(observation, state)
        if output.state is not None:
            options['hidden_state'] = output.state
        action = output.action if self._training else output.mean_action

        options['policy_info'] = {
            "type": self.__class__.__name__,
            "distribution": "gaussian_tanh",
            "mean_raw": output.mean.detach().cpu().tolist(),
            "mean_action": output.mean_action.detach().cpu().tolist(),
            "std": output.std.detach().cpu().tolist(),
            "log_prob": output.log_prob.detach().cpu().tolist(),
            "raw_action": output.raw_action.detach().cpu().tolist(),
            "action": action.detach().cpu().tolist(),
        }
        return action, options

    def clone(self) -> "StochasticPolicy":
        return StochasticPolicy(actor=self._actor.clone())

    def share_memory(self) -> "StochasticPolicy":
        self._actor.share_memory()
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._actor.state_dict()

    def load_state_dict(self, state_dict):
        return self._actor.load_state_dict(state_dict)

    def train(self, mode: bool = True) -> "StochasticPolicy":
        self._training = mode
        self._actor.train(mode)
        return self

    def eval(self) -> "StochasticPolicy":
        return self.train(False)

    def warmup(self, observation_space: Box, num_steps: int = 100) -> None:
        self._actor.eval()
        with torch.no_grad():
            for _ in range(num_steps):
                fake_obs = observation_space.sample()
                self._actor(fake_obs)
