import os
import torch
from torchrl.data import UnboundedContinuous, BoundedContinuous
from torchrl.envs import TransformedEnv, EnvBase

from nn_laser_stabilizer.envs.pid_tuning_env import PidTuningEnv
from nn_laser_stabilizer.connection import create_connection_to_pid
from nn_laser_stabilizer.envs.experimental_setup_controller import ExperimentalSetupController
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
    
    pid_connection = create_connection_to_pid(config, output_dir)
    
    warmup_steps = env_config.warmup_steps
    block_size = env_config.block_size
    max_buffer_size = max(warmup_steps, block_size)
    
    setup_controller = ExperimentalSetupController(
        pid_connection=pid_connection,
        setpoint=env_config.setpoint,
        warmup_steps=warmup_steps,
        block_size=block_size,
        max_buffer_size=max_buffer_size,
        force_min_value=env_config.force_min_value,
        force_max_value=env_config.force_max_value,
        default_min=env_config.default_min,
        default_max=env_config.default_max,
    )
    
    specs = make_specs(env_config.bounds)
    
    env_log_dir = os.path.join(output_dir, "env_logs")
    env_logger = AsyncFileLogger(log_dir=env_log_dir, filename="env.log")
    
    env = PidTuningEnv(
        setup_controller=setup_controller,
        action_spec=specs["action"],
        observation_spec=specs["observation"], 
        reward_spec=BoundedContinuous(low=-1, high=1, shape=(1,)),
        reward_func=make_reward(config),
        logger=env_logger,
        pretrain_blocks=env_config.pretrain_blocks,
        burn_in_steps=env_config.burn_in_steps,
    )
    env.set_seed(config.seed)
    return env
     
def close_real_env(env: TransformedEnv):
    try:
        env.base_env.setup_controller.close()
    except Exception as e:
        print(f"Warning: Could not close serial connection properly: {e}")