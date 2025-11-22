import torch


class ReplayBuffer:
    """
    Буфер повторений для хранения опыта.
    
    Ожидается, что будет лишь один поток-читатель и один поток-писатель.
    Тензоры всегда на CPU.
    """
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int
    ):
        self.capacity = capacity
        
        self._size = torch.zeros(1, dtype=torch.int32, device='cpu')
        self._index = torch.zeros(1, dtype=torch.int32, device='cpu')
        
        self._observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device='cpu')
        self._actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device='cpu')
        self._rewards = torch.zeros((capacity, 1), dtype=torch.float32, device='cpu')
        self._next_observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device='cpu')
        self._dones = torch.zeros((capacity, 1), dtype=torch.bool, device='cpu')
    
    @property
    def observations(self) -> torch.Tensor:
        return self._observations
    
    @property
    def actions(self) -> torch.Tensor:
        return self._actions
    
    @property
    def rewards(self) -> torch.Tensor:
        return self._rewards
    
    @property
    def next_observations(self) -> torch.Tensor:
        return self._next_observations
    
    @property
    def dones(self) -> torch.Tensor:
        return self._dones
    
    def share_memory(self) -> None:
        self._size.share_memory_()
        self._index.share_memory_()
        self._observations.share_memory_()
        self._actions.share_memory_()
        self._rewards.share_memory_()
        self._next_observations.share_memory_()
        self._dones.share_memory_()
    
    @property
    def size(self) -> int:
        return int(self._size[0].item())
    
    @size.setter
    def size(self, value: int) -> None:
        self._size[0] = value
    
    @property
    def index(self) -> int:
        return int(self._index[0].item())
    
    @index.setter
    def index(self, value: int) -> None:
        self._index[0] = value
    
    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
    ) -> None: 
        """
        Ожидается, что тензоры будут отсоединены от графа вычислений.
        """
        self._observations[self.index].copy_(observation)
        self._actions[self.index].copy_(action)
        self._rewards[self.index].copy_(reward)
        self._next_observations[self.index].copy_(next_observation)
        self._dones[self.index].copy_(done)
        
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def __len__(self) -> int:
        return self.size

