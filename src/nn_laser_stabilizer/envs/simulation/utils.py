from nn_laser_stabilizer.envs.simulation.partial_observed_envs import PendulumNoVelEnv
from nn_laser_stabilizer.envs.transforms import FrameSkipTransform

from torchrl.envs import Compose, DoubleToFloat, GymEnv, StepCounter, TransformedEnv


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