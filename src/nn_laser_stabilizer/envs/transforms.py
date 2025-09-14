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


class StepsAggregateTransform(Transform):
    def __init__(
        self,
        frame_skip: int = 1,
        obs_aggregate: str = "mean",
        reward_aggregate: str = "mean",
    ):
        super().__init__()
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1.")
        if obs_aggregate not in ("last", "mean", "sum"):
            raise ValueError("obs_aggregate must be one of: 'last', 'mean', 'sum'.")
        if reward_aggregate not in ("last", "mean", "sum"):
            raise ValueError("reward_aggregate must be one of: 'last', 'mean', 'sum'.")

        self.frame_skip = frame_skip
        self.obs_aggregate = obs_aggregate
        self.reward_aggregate = reward_aggregate

        self._initialized = False
        self._rewards_buf = None
        self._obs_buf = None
        self._reward_key = None
        self._obs_key = None

    def _initialize(self, parent, next_tensordict):
        # TODO: переделать когда-нибудь
        self._reward_key = "reward"
        self._obs_key = "observation" # предполагается один ключ observation

        obs_shape = next_tensordict.get(self._obs_key).shape
        reward_shape = next_tensordict.get(self._reward_key).shape

        self._rewards_buf = torch.zeros((self.frame_skip,) + reward_shape)
        self._obs_buf = torch.zeros((self.frame_skip,) + obs_shape)

        self._initialized = True

    def _aggregate(self, buf, how: str):
        if how == "last":
            return buf[-1]
        elif how == "mean":
            return torch.mean(buf, dim=0)
        elif how == "sum":
            return torch.sum(buf, dim=0)
        else:
            raise RuntimeError(f"Unknown aggregation method: {how}")

    def _step(self, tensordict, next_tensordict):
        parent = self.parent
        if not self._initialized:
            self._initialize(parent, next_tensordict)

        self._rewards_buf[0].copy_(next_tensordict.get(self._reward_key))
        self._obs_buf[0].copy_(next_tensordict.get(self._obs_key))

        for i in range(1, self.frame_skip):
            next_tensordict = parent._step(tensordict)
            self._rewards_buf[i].copy_(next_tensordict.get(self._reward_key))
            self._obs_buf[i].copy_(next_tensordict.get(self._obs_key))

        reward = self._aggregate(self._rewards_buf, self.reward_aggregate)
        obs = self._aggregate(self._obs_buf, self.obs_aggregate)

        return next_tensordict.set(self._reward_key, reward).set(self._obs_key, obs)

    def forward(self, tensordict):
        raise RuntimeError(
            "FrameStackAggregateTransform can only be used when appended to a transformed env."
        )
