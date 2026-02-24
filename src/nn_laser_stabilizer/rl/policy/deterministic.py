from typing import Any

import torch

from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.model.deterministic_actor import DeterministicActor, DeterministicActorOutput
from nn_laser_stabilizer.rl.policy.policy import Policy


class DeterministicPolicy(Policy):
    def __init__(self, actor: DeterministicActor):
        self._actor = actor

    @torch.no_grad()
    def act(self, observation: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        state = options.get('hidden_state')
        output : DeterministicActorOutput = self._actor(observation, state)
        if output.state is not None:
            options['hidden_state'] = output.state
        options['policy_info'] = {
            "distribution": "deterministic",
            "action": output.action.detach().cpu().tolist(),
        }
        return output.action, options

    def clone(self) -> "DeterministicPolicy":
        return DeterministicPolicy(actor=self._actor.clone())

    def share_memory(self) -> "DeterministicPolicy":
        self._actor.share_memory()
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._actor.state_dict()

    def load_state_dict(self, state_dict):
        return self._actor.load_state_dict(state_dict)

    def train(self, mode: bool = True) -> "DeterministicPolicy":
        self._actor.train(mode)
        return self

    def eval(self) -> "DeterministicPolicy":
        self._actor.eval()
        return self

    def warmup(self, observation_space: Box, num_steps: int = 100) -> None:
        self._actor.eval()
        with torch.no_grad():
            for _ in range(num_steps):
                fake_obs = observation_space.sample()
                self._actor(fake_obs)
