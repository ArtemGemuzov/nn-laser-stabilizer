import torch
from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec
from torchrl.envs import EnvBase, GymEnv


class PendulumNoVelEnv(EnvBase):
    def __init__(self):
        super().__init__()

        self.env = GymEnv("Pendulum-v1")

        self.action_spec = self.env.action_spec
        self.reward_spec = self.env.reward_spec

        self.observation_spec = BoundedTensorSpec(
            shape=torch.Size([2]),
            dtype=torch.float32,
            minimum=-1.0,
            maximum=1.0,
        )

    def _filter_observation(self, full_obs: torch.Tensor) -> torch.Tensor:
        # Оставляем только cos(theta) и sin(theta)
        return full_obs[..., :2]  # [cos(theta), sin(theta)]

    def _step(self, tensordict: TensorDict) -> TensorDict:
        td_out = self.env._step(tensordict)

        obs = td_out.get("observation")
        td_out.set("observation", self._filter_observation(obs))

        return td_out

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        td_out = self.env._reset(tensordict)

        obs = td_out.get("observation")
        td_out.set("observation", self._filter_observation(obs))

        return td_out

    def _set_seed(self, seed: int):
        self.env.set_seed(seed)

    def set_state(self, state):
        return self.env.set_state(state)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        if "observation" not in tensordict:
            return self.reset()
        return self.step(tensordict)
