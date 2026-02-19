from typing import Optional

import gymnasium as gym

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.envs.envs import CUSTOM_ENV_MAP
from nn_laser_stabilizer.rl.envs.torch_wrapper import TorchEnvWrapper
from nn_laser_stabilizer.rl.envs.wrappers.step_tracking import StepTrackingWrapper
from nn_laser_stabilizer.rl.envs.wrappers.info_logging import InfoLoggingWrapper
from nn_laser_stabilizer.rl.envs.wrappers.reward_ema import RewardEMAWrapper
from nn_laser_stabilizer.utils.logger import AsyncFileLogger, Logger, NoOpLogger


def _create_logger(wrappers_config: Config | None) -> Logger:
    if wrappers_config is None:
        return NoOpLogger()
    info_logging = wrappers_config.get("info_logging", None)
    if not info_logging or info_logging is True:
        return NoOpLogger()
    return AsyncFileLogger(
        log_dir=info_logging.log_dir,
        log_file=info_logging.log_file,
    )


def _apply_wrappers(
    env: gym.Env, wrappers_config: Config | None, logger: Logger
) -> gym.Env:
    if wrappers_config is None:
        return env

    step_tracking = wrappers_config.get("step_tracking", None)
    if step_tracking is not None:
        time_multiplier = 1e6
        if step_tracking is not True:
            time_multiplier = float(step_tracking.get("time_multiplier", 1e6))
        env = StepTrackingWrapper(env, time_multiplier=time_multiplier)

    reward_ema = wrappers_config.get("reward_ema", None)
    if reward_ema is not None:
        alpha = 0.99
        if reward_ema is not True:
            alpha = float(reward_ema.get("alpha", 0.3))
        env = RewardEMAWrapper(env, alpha=alpha)

    info_logging = wrappers_config.get("info_logging", None)
    if info_logging is not None:
        env = InfoLoggingWrapper(env, logger)

    return env


def make_env_from_config(env_config: Config, seed: Optional[int] = None) -> TorchEnvWrapper:
    wrappers_config = env_config.get("wrappers", None)
    logger = _create_logger(wrappers_config)

    env_name = env_config.name
    if env_name in CUSTOM_ENV_MAP:
        env_class = CUSTOM_ENV_MAP[env_name]
        env = env_class.from_config(env_config, logger)
    else:
        args = env_config.get("args")
        env_kwargs = args.to_dict() if args is not None else {}
        try:
            env = gym.make(env_name, **env_kwargs)
        except gym.error.UnregisteredEnv:
            raise ValueError(
                f"Unknown environment: '{env_name}'. "
                f"Custom environments: {list(CUSTOM_ENV_MAP.keys())}"
            )

    env = _apply_wrappers(env, wrappers_config, logger)
    return TorchEnvWrapper.wrap(env, seed=seed)


def get_spaces_from_config(
    env_config: Config,
    seed: Optional[int] = None,
) -> tuple[Box, Box]:
    env = make_env_from_config(env_config, seed=seed)
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    return observation_space, action_space
