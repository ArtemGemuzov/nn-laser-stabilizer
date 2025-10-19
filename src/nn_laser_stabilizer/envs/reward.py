import inspect
import numpy as np

from omegaconf import DictConfig
from nn_laser_stabilizer.envs.normalizer import Normalizer

class AbsoluteErrorReward:
    """
    Награда: -|error_norm| + 1
    """
    def __init__(self, normalizer: Normalizer):
        self.normalizer = normalizer

    def __call__(self, process_variable, setpoint):
        """
        Args:
            process_variable: float или np.ndarray
            setpoint: float или np.ndarray
            
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

    def __call__(self, process_variable, setpoint):
        """
        Args:
            process_variable: float или np.ndarray
            setpoint: float или np.ndarray
            
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

    def __call__(self, process_variable, setpoint):
        """
        Args:
            process_variable: float или np.ndarray
            setpoint: float или np.ndarray
            
        Returns:
            float или np.ndarray (в зависимости от входа)
        """
        relative_error = np.abs(setpoint - process_variable) / (np.abs(setpoint) + self.eps)
        reward = 1 - relative_error
        return np.maximum(-1.0, reward)
    

REWARD_FUNCTIONS = {
    "absolute": AbsoluteErrorReward,
    "relative": RelativeErrorReward,
    "exponential": ExponentialErrorReward,
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
 