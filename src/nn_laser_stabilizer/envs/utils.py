from torchrl.envs import TransformedEnv, DoubleToFloat, Compose, InitTracker, GymEnv, StepCounter

from nn_laser_stabilizer.envs.pid_controller import PIDController
from nn_laser_stabilizer.envs.oscillator import DuffingOscillator
from nn_laser_stabilizer.envs.numerical_experimental_setup import NumericalExperimentalSetup
from nn_laser_stabilizer.envs.pid_tuning_experimental_env import PidTuningExperimentalEnv

def make_env(config) -> TransformedEnv:
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

    if False:
        base_env = GymEnv("Pendulum-v1")
    else:
        base_env = PidTuningExperimentalEnv(numerical_model)
    env = TransformedEnv(
        base_env,
        Compose(
            StepCounter(),
            InitTracker(),
            DoubleToFloat()
        )
    )
    env.set_seed(config.seed)
    return env