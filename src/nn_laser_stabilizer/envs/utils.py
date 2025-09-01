from torchrl.envs import TransformedEnv, DoubleToFloat, Compose, InitTracker

from nn_laser_stabilizer.envs.pid_controller import PIDController
from nn_laser_stabilizer.envs.oscillator import DuffingOscillator
from nn_laser_stabilizer.envs.numerical_experimental_setup import NumericalExperimentalSetup
from nn_laser_stabilizer.envs.pid_tuning_experimental_env import PidTuningExperimentalEnv
from nn_laser_stabilizer.serial_connection import SerialConnection
from nn_laser_stabilizer.mock_serial_connection import MockSerialConnection
from nn_laser_stabilizer.envs.real_experimental_setup import RealExperimentalSetup

from torchrl.envs import GymEnv
from torchrl.envs.transforms import Transform, Compose, DoubleToFloat, ObservationNorm, RewardScaling
from torchrl.envs.transforms import StepCounter, InitTracker
from torchrl.data import UnboundedContinuous, BoundedContinuous
import torch

from torch.utils.tensorboard import SummaryWriter

class PerStepLogger(Transform):
    """
    Логирует kp, ki, kd (из action) и x, setpoint (из observation) каждый шаг.
    """
    def __init__(self, writer=None):
        super().__init__()
        self.writer = writer
        self._t = 0 

    def _log_step(self, action_row: torch.Tensor, observation_row: torch.Tensor):
        kp = action_row[0].item()
        ki = action_row[1].item()
        kd = action_row[2].item()

        x = observation_row[0].item()
        setpoint = observation_row[2].item()

        self.writer.add_scalar("Action/kp", kp, self._t)
        self.writer.add_scalar("Action/ki", ki, self._t)
        self.writer.add_scalar("Action/kd", kd, self._t)
        self.writer.add_scalar("Observation/x", x, self._t)
        self.writer.add_scalar("Observation/setpoint", setpoint, self._t)

        self._t += 1

    def _step(self, tensordict, next_tensordict):
        """
        Вызывается на каждом env.step().
        tensordict: данные на шаге t
        next_tensordict: данные на шаге t+1
        """      
        action = tensordict.get("action", None)
        observation = tensordict.get("observation", None)

        if action is not None and observation is not None:
            self._log_step(action, observation)

        return next_tensordict

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

    base_env = GymEnv(env_config.name)
    env = TransformedEnv(
        base_env,
        Compose(
            InitTracker(),
            StepCounter(),
            DoubleToFloat(),
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
        reward_spec=specs["reward"]
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
    
    serial_connection = SerialConnection(
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
        reward_spec=BoundedContinuous(low=-1, high=1, shape=(1,))
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
    writer = SummaryWriter(log_dir=logdir)
    env.append_transform(PerStepLogger(writer=writer))
    return env