import os
import torch
from torchrl.data import UnboundedContinuous, BoundedContinuous
from torchrl.envs import TransformedEnv, EnvBase

from nn_laser_stabilizer.envs.pid_tuning_experimental_env import PidTuningExperimentalEnv
from nn_laser_stabilizer.connection import create_connection, ConnectionToPid, LoggingConnectionToPid
from nn_laser_stabilizer.envs.real_experimental_setup import RealExperimentalSetup
from nn_laser_stabilizer.envs.reward import make_reward
from nn_laser_stabilizer.logging.async_file_logger import AsyncFileLogger

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

def make_real_env(config, output_dir: str) -> EnvBase:
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
            - serial.log_connection: логировать ли команды и ответы (опционально, по умолчанию False)
            - seed: зерно для генератора случайных чисел
        output_dir: Рабочая директория для логов
    
    Returns:
        TransformedEnv: Окружение TorchRL
    """
    env_config = config.env
    
    serial_connection = create_connection(config)
    serial_connection.open_connection()
    
    pid_connection = ConnectionToPid(serial_connection)
    
    if config.serial.log_connection:
        connection_log_dir = os.path.join(output_dir, "connection_logs")
        connection_logger = AsyncFileLogger(log_dir=connection_log_dir, filename="connection.log")
        pid_connection = LoggingConnectionToPid(pid_connection, connection_logger)
    
    real_setup = RealExperimentalSetup(
        pid_connection=pid_connection,
        setpoint=env_config.setpoint
    )
    
    specs = make_specs(env_config.bounds)
    
    env_log_dir = os.path.join(output_dir, "env_logs")
    env_logger = AsyncFileLogger(log_dir=env_log_dir, filename="env.log")
    
    env = PidTuningExperimentalEnv(
        real_setup,
        action_spec=specs["action"],
        observation_spec=specs["observation"], 
        reward_spec=BoundedContinuous(low=-1, high=1, shape=(1,)),
        reward_func=make_reward(config),
        logger=env_logger,
        warmup_steps=env_config.get('warmup_steps', 1000),
        pretrain_blocks=env_config.get('pretrain_blocks', 100),
        block_size=env_config.get('block_size', 100),
        burn_in_steps=env_config.get('burn_in_steps', 20),
        force_min_value=env_config.get('force_min_value', 2000.0),
        force_max_value=env_config.get('force_max_value', 4095.0),
        default_min=env_config.get('default_min', 0.0),
        default_max=env_config.get('default_max', 4095.0),
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