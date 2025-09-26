import torch
from torchrl.data import UnboundedContinuous, BoundedContinuous
from torchrl.envs import TransformedEnv, EnvBase, Compose

from nn_laser_stabilizer.envs.logger import PerStepLoggerAsync
from nn_laser_stabilizer.envs.pid_tuning_experimental_env import PidTuningExperimentalEnv
from nn_laser_stabilizer.connection.serial_connection import SerialConnection
from nn_laser_stabilizer.connection.mock_serial_connection import MockSerialConnection
from nn_laser_stabilizer.envs.real_experimental_setup import RealExperimentalSetup
from nn_laser_stabilizer.envs.reward import make_reward
from nn_laser_stabilizer.envs.transforms import InitialActionRepeatTransform, StepsAggregateTransform
from nn_laser_stabilizer.envs.logger import LoggingEnvWrapper

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

def make_real_env(config) -> EnvBase:
    """
    Создает окружение TorchRL для взаимодействия с реальной установкой через SerialConnection.
    
    Args:
        config: Конфигурация, содержащая:
            - env.setpoint: целевое значение (setpoint)
            - env.bounds: границы для спецификаций
            - serial.use_mock: использовать ли mock соединение (True/False)
            - serial.port: COM порт для подключения
            - serial.baudrate: скорость передачи (опционально, по умолчанию 115200)
            - serial.timeout: таймаут (опционально, по умолчанию 0.1)
            - seed: зерно для генератора случайных чисел
    
    Returns:
        TransformedEnv: Окружение TorchRL
    """
    env_config = config.env
    serial_config = config.serial
    
    if serial_config.use_mock:
        serial_connection = MockSerialConnection(
            port=serial_config.port,
            baudrate=serial_config.baudrate,
            timeout=serial_config.timeout,
        )
    else:
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
    
    fixed_kp = env_config.get('kp', None)
    fixed_ki = env_config.get('ki', None)
    fixed_kd = env_config.get('kd', None)

    control_output_limits_config = env_config.control_output_limits

    env = PidTuningExperimentalEnv(
        real_setup,
        action_spec=specs["action"],
        observation_spec=BoundedContinuous(low=-1, high=1, shape=(3,)),
        reward_spec=BoundedContinuous(low=-1, high=1, shape=(1,)),
        reward_func=make_reward(config),
        fixed_kp=fixed_kp,
        fixed_ki=fixed_ki,
        fixed_kd=fixed_kd,
        default_min=control_output_limits_config.get('default_min', None),
        default_max=control_output_limits_config.get('default_max', None),
        force_min_value=control_output_limits_config.get('force_min_value', None),
        force_condition_threshold=control_output_limits_config.get('force_condition_threshold', None),
        enforcement_steps=control_output_limits_config.get('enforcement_steps', None),
    )
    env.set_seed(config.seed)
    return env

def transform_env(config, base_env):
    env_config = config.env

    transformed_env = TransformedEnv(
        base_env,
        Compose(
            InitialActionRepeatTransform(repeat_count=env_config.repeat_count),
            StepsAggregateTransform(frame_skip=env_config.frame_skip),
        ),
    )
    return transformed_env

def wrap_with_logger(env, log_dir):
    return LoggingEnvWrapper(env, log_dir=log_dir)
     
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