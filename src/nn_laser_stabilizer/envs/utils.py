from torchrl.envs import TransformedEnv, DoubleToFloat, Compose, InitTracker

from nn_laser_stabilizer.envs.pid_controller import PIDController
from nn_laser_stabilizer.envs.oscillator import DuffingOscillator
from nn_laser_stabilizer.envs.numerical_experimental_setup import NumericalExperimentalSetup
from nn_laser_stabilizer.envs.pid_tuning_experimental_env import PidTuningExperimentalEnv

def make_env(config) -> TransformedEnv:
    pid = PIDController(setpoint=config.setpoint)
    oscillator = DuffingOscillator(
        mass=config.mass, 
        k_linear=config.k_linear, 
        k_nonlinear=config.k_nonlinear, 
        k_damping=config.k_damping,
        process_noise_std=config.process_noise_std, 
        measurement_noise_std=config.measurement_noise_std
    )
    numerical_model = NumericalExperimentalSetup(oscillator, pid)

    base_env = PidTuningExperimentalEnv(numerical_model)
    env = TransformedEnv(
        base_env,
        Compose(
            InitTracker(),
            DoubleToFloat()
        )
    )
    env.set_seed(config.seed)
    return env