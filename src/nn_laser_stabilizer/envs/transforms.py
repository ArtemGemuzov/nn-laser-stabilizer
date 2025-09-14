import torch
from torchrl.envs.transforms import Transform


class FrameSkipTransform(Transform):
    def __init__(self, frame_skip: int = 1):
        super().__init__()
        if frame_skip < 1:
            raise ValueError("frame_skip should be >= 1.")
        self.frame_skip = frame_skip

    def _aggregate_rewards(self, rewards):
        return torch.sum(rewards)

    def _step(self, tensordict, next_tensordict):
        parent = self.parent
        if parent is None:
            raise RuntimeError("Parent environment not found.")
        reward_key = parent.reward_key

        rewards = torch.zeros(self.frame_skip, device=next_tensordict.get(reward_key).device)
        rewards[0] = next_tensordict.get(reward_key)

        for i in range(1, self.frame_skip):
            next_tensordict = parent._step(tensordict)
            rewards[i] = next_tensordict.get(reward_key)

        reward = self._aggregate_rewards(rewards)
        return next_tensordict.set(reward_key, reward)

    def forward(self, tensordict):
        raise RuntimeError(
            "FrameSkipAverageRewardTransform can only be used when appended to a transformed env."
        )
    
    
class InitialActionRepeatTransform(Transform):
    def __init__(self, repeat_count: int = 1):
        super().__init__()
        if repeat_count < 1:
            raise ValueError("repeat_count must be >= 1.")
        self.repeat_count = repeat_count
        self._initialized = False

    def _step(self, tensordict, next_tensordict):
        parent = self.parent
        if parent is None:
            raise RuntimeError("Parent environment not found.")

        if not self._initialized:
            for _ in range(1, self.repeat_count):
                next_tensordict = parent._step(tensordict)
            self._initialized = True
        return next_tensordict

    def forward(self, tensordict):
        raise RuntimeError(
            "InitialActionRepeatTransform can only be used when appended to a transformed env."
        )