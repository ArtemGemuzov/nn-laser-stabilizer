import torch
import gymnasium as gym


class Discrete:
    def __init__(self, n: int):
        self.n = n
        self.dim = 1

    @classmethod
    def from_gymnasium(cls, gym_discrete: gym.spaces.Discrete) -> "Discrete":
        return cls(n=int(gym_discrete.n))

    def sample(self) -> torch.Tensor:
        return torch.randint(0, self.n, (1,)).float()
