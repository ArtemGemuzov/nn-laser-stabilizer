import inspect
from typing import Protocol, Union
import numpy as np

from omegaconf import DictConfig
from nn_laser_stabilizer.envs.normalizer import Normalizer


class RewardFunction(Protocol):
    def __call__(self, process_variable: Union[float, np.ndarray], 
                 setpoint: Union[float, np.ndarray], 
                 action: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pass

class AbsoluteErrorReward:
    """
    Награда: -|error_norm| + 1
    """
    def __init__(self, normalizer: Normalizer):
        self.normalizer = normalizer

    def __call__(self, process_variable, setpoint, action):
        """
        Args:
            process_variable: float или np.ndarray
            setpoint: float или np.ndarray
            action: float или np.ndarray
            
        Returns:
            float или np.ndarray (в зависимости от входа)
        """
        process_variable_norm = self.normalizer.normalize_process_variable(process_variable)
        setpoint_norm = self.normalizer.normalize_process_variable(setpoint)

        error = setpoint_norm - process_variable_norm
        reward = -np.abs(error)
        reward_norm = 1 + reward
        return reward_norm
    

class ExponentialErrorReward:
    """
    Награда вычисляется по формуле:

        reward = 2 * exp(-k * |setpoint_norm - process_variable_norm|) - 1

    Где:
      - reward ∈ [-1, 1]
      - k > 0 задаёт жёсткость штрафа
    """
    def __init__(self, normalizer: Normalizer, k: float = 10.0):
        self.normalizer = normalizer
        self.k = k

    def __call__(self, process_variable, setpoint, action):
        """
        Args:
            process_variable: float или np.ndarray
            setpoint: float или np.ndarray
            action: float или np.ndarray
            
        Returns:
            float или np.ndarray (в зависимости от входа)
        """
        process_variable_norm = self.normalizer.normalize_process_variable(process_variable)
        setpoint_norm = self.normalizer.normalize_process_variable(setpoint)
        error = np.abs(setpoint_norm - process_variable_norm)
        return 2 * np.exp(-self.k * error) - 1

class RelativeErrorReward:
    """
    Награда на основе относительной ошибки:

        relative_error = |setpoint - process_variable| / (|setpoint| + eps)
        reward = max{1 - relative_error; -1}

    Где:
      - reward ∈ [-1, 1]
      - eps нужен для защиты от деления на ноль
    """
    def __init__(self, normalizer: Normalizer, eps: float = 1e-6):
        self.normalizer = normalizer
        self.eps = eps

    def __call__(self, process_variable, setpoint, action):
        """
        Args:
            process_variable: float или np.ndarray
            setpoint: float или np.ndarray
            action: float или np.ndarray
            
        Returns:
            float или np.ndarray (в зависимости от входа)
        """
        relative_error = np.abs(setpoint - process_variable) / (np.abs(setpoint) + self.eps)
        reward = 1 - relative_error
        return np.maximum(-1.0, reward)


class ExponentialErrorActionPenaltyReward:
    """
    Награда, объединяющая экспоненциальную награду за ошибку с штрафом за действия:

        error_reward = 2 * exp(-k_error * |setpoint_norm - process_variable_norm|) - 1
        action_penalty = -k_action * |action_norm|
        reward = error_reward + action_penalty

    Где:
      - error_reward ∈ [-1, 1] - экспоненциальная награда за точность
      - action_penalty ≤ 0 - штраф за большие действия
      - k_error > 0 - жёсткость штрафа за ошибку
      - k_action > 0 - вес штрафа за действия
    """
    def __init__(self, normalizer: Normalizer, k_error: float = 10.0, k_action: float = 0.1):
        self.normalizer = normalizer
        self.k_error = k_error
        self.k_action = k_action

    def __call__(self, process_variable, setpoint, action):
        """
        Args:
            process_variable: float или np.ndarray
            setpoint: float или np.ndarray
            action: float или np.ndarray
            
        Returns:
            float или np.ndarray (в зависимости от входа)
        """
        # Нормализация переменной процесса и заданного значения
        process_variable_norm = self.normalizer.normalize_process_variable(process_variable)
        setpoint_norm = self.normalizer.normalize_process_variable(setpoint)
        
        # Вычисление экспоненциальной награды за ошибку
        error = np.abs(setpoint_norm - process_variable_norm)
        error_reward = 2 * np.exp(-self.k_error * error) - 1
        
        # Нормализация действия и вычисление штрафа
        action_norm = self.normalizer.normalize_action(action)
        action_penalty = -self.k_action * np.abs(action_norm)
        
        # Итоговая награда
        return error_reward + action_penalty
    

REWARD_FUNCTIONS = {
    "absolute": AbsoluteErrorReward,
    "relative": RelativeErrorReward,
    "exponential": ExponentialErrorReward,
    "exponential_action_penalty": ExponentialErrorActionPenaltyReward,
}

def make_reward(config: DictConfig, normalizer: Normalizer):
    reward_config = config.env.reward
    
    reward_name = reward_config.name
    if reward_name not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward function: {reward_name}")

    RewardClass = REWARD_FUNCTIONS[reward_name]

    args = reward_config.get("args", None)
    if args is None:
        args = {}

    sig = inspect.signature(RewardClass.__init__)
    valid_args = {k: v for k, v in args.items() if k in sig.parameters and k != "self"}
    
    valid_args["normalizer"] = normalizer
    
    return RewardClass(**valid_args)
 