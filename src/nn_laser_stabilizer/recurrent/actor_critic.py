import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhScaler(nn.Module):
    def __init__(self, min_val: float, max_val: float):
        super(TanhScaler, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tanh(x)
        return self.min_val + (x + 1) * (self.max_val - self.min_val) / 2

class MLPTanhActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size=256, max_action=1.0, min_action=-1.0):
        super(MLPTanhActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        self.scaler = TanhScaler(min_val=min_action, max_val=max_action)

    def forward(self, observation_summary):
        return self.scaler(self.net(observation_summary))

class MLPCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size=256):
        super(MLPCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, observation_summary, action):
        return self.net(torch.cat([observation_summary, action], dim=-1))