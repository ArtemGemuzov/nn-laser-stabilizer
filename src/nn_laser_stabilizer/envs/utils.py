from torchrl.envs import TransformedEnv, DoubleToFloat, Compose, InitTracker

from nn_laser_stabilizer.envs.logger import PerStepLoggerAsync
from nn_laser_stabilizer.envs.pid_controller import PIDController
from nn_laser_stabilizer.envs.oscillator import DuffingOscillator
from nn_laser_stabilizer.envs.numerical_experimental_setup import NumericalExperimentalSetup
from nn_laser_stabilizer.envs.pid_tuning_experimental_env import PidTuningExperimentalEnv
from nn_laser_stabilizer.serial_connection import SerialConnection
from nn_laser_stabilizer.mock_serial_connection import MockSerialConnection
from nn_laser_stabilizer.envs.real_experimental_setup import RealExperimentalSetup
from nn_laser_stabilizer.envs.partial_observed_envs import PendulumNoVelEnv
from nn_laser_stabilizer.envs.reward import make_reward

from torchrl.envs import GymEnv
from torchrl.envs.transforms import Compose, DoubleToFloat, StepCounter, InitTracker, Transform
from torchrl.envs.transforms import StepCounter, InitTracker
from torchrl.data import UnboundedContinuous, BoundedContinuous
import torch

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
    
def make_specs(bounds_config: dict) -> dict:
    specs = {}
    for key in ["action", "observation", "reward"]:
        spec_bounds = bounds_config.get(key)
        if spec_bounds is None:
            raise ValueError(f"Missing bounds for {key}")

        low = torch.tensor([float(x) for x in spec_bounds["low"]])
        high = torch.tensor([float(x) for x in spec_bounds["high"]])

        if torch.isinf(low).any() or torch.isinf(high).any():
            specs[key] = UnboundedContinuous(shape=low.shape)
        else:
            specs[key] = BoundedContinuous(low=low, high=high, shape=low.shape)

    return specs

def make_gym_env(config) -> TransformedEnv:
    env_config = config.env

    env_name = env_config.name
    if env_name == "PendulumNoVel":
        base_env = PendulumNoVelEnv()
    else: 
        base_env = GymEnv(env_config.name)

    env = TransformedEnv(
        base_env,
        Compose(
            StepCounter(),
            DoubleToFloat(),
            FrameSkipTransform(frame_skip=env_config.frame_skip),
        )
    )
    env.set_seed(config.seed)
    return env

def make_simulated_env(config) -> TransformedEnv:
    env_config = config.env

    pid = PIDController(setpoint=env_config.setpoint)
    oscillator = DuffingOscillator(
        mass=env_config.mass, 
        k_linear=env_config.k_linear, 
        k_nonlinear=env_config.k_nonlinear, 
        k_damping=env_config.k_damping,
        process_noise_std=env_config.process_noise_std, 
        measurement_noise_std=env_config.measurement_noise_std
    )
    numerical_model = NumericalExperimentalSetup(oscillator, pid)

    specs = make_specs(env_config.bounds)

    base_env = PidTuningExperimentalEnv(
        numerical_model, 
        action_spec=specs["action"],
        observation_spec=specs["observation"],
        reward_spec=specs["reward"],
        reward_func=make_reward(config)
    )
    
    env = TransformedEnv(
        base_env,
        Compose(
            DoubleToFloat(),
            FrameSkipTransform(frame_skip=env_config.frame_skip),
        )
    )
    env.set_seed(config.seed)
    return env

def make_real_env(config) -> TransformedEnv:
    """
    Создает окружение TorchRL для взаимодействия с реальной установкой через SerialConnection.
    
    Args:
        config: Конфигурация, содержащая:
            - env.setpoint: целевое значение (setpoint)
            - env.bounds: границы для спецификаций
            - serial.port: COM порт для подключения
            - serial.baudrate: скорость передачи (опционально, по умолчанию 115200)
            - serial.timeout: таймаут (опционально, по умолчанию 0.1)
            - seed: зерно для генератора случайных чисел
    
    Returns:
        TransformedEnv: Окружение TorchRL
    """
    env_config = config.env
    serial_config = config.serial
    
    serial_connection = MockSerialConnection(
        port=serial_config.port,
        baudrate=serial_config.baudrate,
        timeout=serial_config.timeout,
    )
    
    serial_connection.open_connection()
    
    real_setup = RealExperimentalSetup(
        serial_connection=serial_connection,
        setpoint=env_config.setpoint
    )
    
    specs = make_specs(env_config.bounds)
    
    base_env = PidTuningExperimentalEnv(
        real_setup,
        action_spec=specs["action"],
        observation_spec=BoundedContinuous(low=-1, high=1, shape=(3,)),
        reward_spec=BoundedContinuous(low=-1, high=1, shape=(1,)),
        reward_func=make_reward(config)
    )
    
    env = TransformedEnv(
        base_env,
        Compose(
            DoubleToFloat(),
            InitialActionRepeatTransform(repeat_count=env_config.repeat_count),
            FrameSkipTransform(frame_skip=env_config.frame_skip),
        )
    )
    
    env.set_seed(config.seed)
    return env


def close_real_env(env: TransformedEnv):
    try:
        real_setup = env.base_env.experimental_setup
        if hasattr(real_setup, 'serial_connection'):
            real_setup.serial_connection.close_connection()
    except Exception as e:
        print(f"Warning: Could not close serial connection properly: {e}")

def add_logger_to_env(env: TransformedEnv, logdir) -> TransformedEnv:
    env.append_transform(PerStepLoggerAsync(log_dir=logdir))
    return env