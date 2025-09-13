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
from torchrl.envs.transforms import Transform, Compose, DoubleToFloat, ObservationNorm, RewardScaling
from torchrl.envs.transforms import StepCounter, InitTracker
from torchrl.data import UnboundedContinuous, BoundedContinuous
from tensordict import TensorDictBase
import torch


class ObservationActionConcat(Transform):
    def __init__(self, obs_key="observation", action_key="action",
                 out_key="observation_action", dim=-1):
        super().__init__(in_keys=[obs_key], out_keys=[out_key])
        self.obs_key = obs_key
        self.action_key = action_key
        self.out_key = out_key
        self.dim = dim

    def _get_zero_action(self, tensordict: TensorDictBase) -> torch.Tensor:
        action_spec = self.parent.action_spec_unbatched
        zero_action = torch.zeros(action_spec.shape,
                                 dtype=action_spec.dtype,
                                 device=tensordict.device,
                                 requires_grad=True)
        return zero_action

    def _step(self, tensordict: TensorDictBase,
              next_tensordict: TensorDictBase) -> TensorDictBase:
        obs = next_tensordict.get(self.obs_key)
        action = tensordict.get(self.action_key)
       
        obs_action = torch.cat([obs, action], dim=self.dim)
        next_tensordict.set(self.out_key, obs_action)
        return next_tensordict

    def _reset(self, tensordict: TensorDictBase,
               tensordict_reset: TensorDictBase) -> TensorDictBase:
        obs = tensordict_reset.get(self.obs_key)
        zero_action = self._get_zero_action(tensordict_reset)
        obs_action = torch.cat([obs, zero_action], dim=self.dim)
        tensordict_reset.set(self.out_key, obs_action)
        return tensordict_reset

    def transform_output_spec(self, output_spec):
        """Добавляем новый ключ 'observation_action' в output_spec."""
        output_spec = output_spec.clone()
        obs_spec = output_spec["full_observation_spec"]

        obs_shape = obs_spec[self.obs_key].shape
        action_spec = self.parent.action_spec_unbatched
        action_shape = action_spec.shape
        dtype = obs_spec[self.obs_key].dtype

        obs_action_spec = UnboundedContinuous(
            shape=(obs_shape[-1] + action_shape[-1]),
            dtype=dtype
        )

        obs_spec[self.out_key] = obs_action_spec
        output_spec["full_observation_spec"] = obs_spec
        return output_spec

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
            InitTracker(),
            StepCounter(),
            DoubleToFloat(),
            ObservationActionConcat(),
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
            InitTracker(),
            DoubleToFloat(),
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
            InitTracker(),
            DoubleToFloat(),
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