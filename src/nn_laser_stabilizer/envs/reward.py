import math
import inspect

from nn_laser_stabilizer.envs.normalization import normalize_adc, normalize_dac
from omegaconf import DictConfig

class AbsoluteErrorReward:
    """
    Награда: -|error_norm| + 1
    """
    def __init__(self):
        pass

    def __call__(self, process_variable: float, setpoint: float) -> float:
        process_variable_norm = normalize_adc(process_variable)
        setpoint_norm = normalize_adc(setpoint)

        error = setpoint_norm - process_variable_norm
        reward = -abs(error)
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
    def __init__(self, k: float = 10.0):
        self.k = k

    def __call__(self, process_variable: float, setpoint: float) -> float:
        process_variable_norm = normalize_adc(process_variable)
        setpoint_norm = normalize_adc(setpoint)
        error = abs(setpoint_norm - process_variable_norm)
        return 2 * math.exp(-self.k * error) - 1
    
REWARD_FUNCTIONS = {
    "absolute": AbsoluteErrorReward,
    "exponential": ExponentialErrorReward,
}

def make_reward(config: DictConfig):
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
    return RewardClass(**valid_args)
 