from nn_laser_stabilizer.envs.constants import DAC_MAX
from nn_laser_stabilizer.envs.pid_tuning_experimental_env import PidTuningExperimentalEnv
from nn_laser_stabilizer.envs.reward import make_reward
from nn_laser_stabilizer.envs.simulation.numerical_experimental_setup import NumericalExperimentalSetup
from nn_laser_stabilizer.envs.simulation.oscillator import DuffingOscillator
from nn_laser_stabilizer.envs.simulation.partial_observed_envs import PendulumNoVelEnv
from nn_laser_stabilizer.envs.simulation.pid_controller import PIDController
from nn_laser_stabilizer.envs.transforms import FrameSkipTransform


from torchrl.envs import Compose, DoubleToFloat, GymEnv, StepCounter, TransformedEnv

from nn_laser_stabilizer.envs.utils import make_specs


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

    fixed_kp = env_config.get('kp', None)
    fixed_ki = env_config.get('ki', None)
    fixed_kd = env_config.get('kd', None)

    control_output_limits_config = getattr(env_config, 'control_output_limits', None)
    if control_output_limits_config is None:
        control_output_limits_config = {
            'default_min': 0.0,
            'default_max': DAC_MAX,
            'force_min_value': 2000.0,
            'force_condition_threshold': 500.0,
            'enforcement_steps': 1000,
        }

    base_env = PidTuningExperimentalEnv(
        numerical_model,
        action_spec=specs["action"],
        observation_spec=specs["observation"],
        reward_spec=specs["reward"],
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

    env = TransformedEnv(
        base_env,
        Compose(
            DoubleToFloat(),
        )
    )
    env.set_seed(config.seed)
    return env