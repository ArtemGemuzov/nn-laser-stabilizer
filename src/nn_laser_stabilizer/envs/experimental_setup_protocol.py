from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np


class ExperimentalSetupProtocol(ABC):
    @abstractmethod
    def step(self, kp: float, ki: float, kd: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
    
    @abstractmethod
    def reset(self, kp: float = None, ki: float = None, kd: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
    
    @abstractmethod
    def set_seed(self, seed: Optional[int]) -> None:
        pass
    
    @abstractmethod
    def close(self) -> None:
        pass
    
    @property
    @abstractmethod
    def setpoint(self) -> float:
        pass
