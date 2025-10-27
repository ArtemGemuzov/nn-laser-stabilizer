import os
import torch
from torchrl.data import UnboundedContinuous, BoundedContinuous
from torchrl.envs import TransformedEnv, EnvBase

from nn_laser_stabilizer.envs.pid_tuning_env import PidTuningEnv, PidDeltaTuningEnv
from nn_laser_stabilizer.connection import create_connection_to_pid
from nn_laser_stabilizer.envs.experiment.experimental_setup_controller import ExperimentalSetupController
from nn_laser_stabilizer.envs.simulation.numerical_experimental_setup_controller import NumericalExperimentalSetupController
from nn_laser_stabilizer.envs.simulation.oscillator import DuffingOscillator
from nn_laser_stabilizer.envs.simulation.first_order_system import FirstOrderSystem
from nn_laser_stabilizer.envs.simulation.pid_controller import PIDController
from nn_laser_stabilizer.envs.reward import make_reward
from nn_laser_stabilizer.envs.normalizer import Normalizer, make_normalizer
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

def make_env(config) -> EnvBase:
    env_config = config.env
    env_name = env_config.name
    
    normalizer = make_normalizer(config)
    
    if env_name == "real":
        return _make_real_env(config, normalizer)
    elif env_name == "linear":
        return _make_linear_env(config, normalizer)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")

def _make_real_env(config, normalizer: Normalizer) -> EnvBase:
    env_config = config.env
    output_dir = config.output_dir
    
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
    
    env = PidDeltaTuningEnv(
        setup_controller=setup_controller,
        action_spec=specs["action"],
        observation_spec=specs["observation"], 
        reward_spec=BoundedContinuous(low=-1, high=1, shape=(1,)),
        reward_func=make_reward(config, normalizer),
        normalizer=normalizer,
        logger=env_logger,
        pretrain_blocks=env_config.pretrain_blocks,
        burn_in_steps=env_config.burn_in_steps,
    )
    env.set_seed(config.seed)
    return env


def _make_linear_env(config, normalizer: Normalizer) -> EnvBase:
    env_config = config.env
    output_dir = config.output_dir

    setup_controller = NumericalExperimentalSetupController(
        plant=FirstOrderSystem(
            time_constant=env_config.time_constant,
            gain=env_config.gain,
            initial_process_variable=env_config.initial_process_variable,
        ),
        setpoint=env_config.setpoint,
        warmup_steps=env_config.warmup_steps,
        block_size=env_config.block_size,
        dt=env_config.dt,
    )

    specs = make_specs(env_config.bounds)

    env_log_dir = os.path.join(output_dir, "env_logs")
    env_logger = AsyncFileLogger(log_dir=env_log_dir, filename="env.log")

    env = PidTuningEnv(
        setup_controller=setup_controller,
        action_spec=specs["action"],
        observation_spec=specs["observation"], 
        reward_spec=BoundedContinuous(low=-1, high=1, shape=(1,)),
        reward_func=make_reward(config, normalizer),
        normalizer=normalizer,
        logger=env_logger,
        pretrain_blocks=env_config.pretrain_blocks,
        burn_in_steps=env_config.burn_in_steps,
    )
    env.set_seed(config.seed)
    return env
     
def close_env(env: TransformedEnv):
    try:
        env.base_env.setup_controller.close()
    except Exception as e:
        print(f"Warning: Could not close serial connection properly: {e}")