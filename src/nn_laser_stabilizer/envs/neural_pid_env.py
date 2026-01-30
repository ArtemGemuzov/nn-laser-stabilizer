from typing import Optional

import numpy as np
import gymnasium as gym

from nn_laser_stabilizer.logger import AsyncFileLogger, Logger, PrefixedLogger
from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.envs.base_env import BaseEnv
from nn_laser_stabilizer.envs.neural_controller_phys import NeuralControllerPhys
from nn_laser_stabilizer.normalize import (
    denormalize_from_minus1_plus1,
    normalize_to_minus1_plus1,
    normalize_to_01,
)


class NeuralPIDEnv(BaseEnv):
    LOG_PREFIX = "ENV"

    def __init__(
        self,
        *,
        physics: NeuralControllerPhys,
        base_logger: Logger,
        control_min: int,
        control_max: int,
        process_variable_max: float,
    ):
        super().__init__()

        self._control_min = int(control_min)
        self._control_max = int(control_max)
        self._process_variable_max = float(process_variable_max)

        self._base_logger = base_logger
        self._env_logger = PrefixedLogger(self._base_logger, NeuralPIDEnv.LOG_PREFIX)
        self._physics = physics

        self._step: int = 0

        self._setpoint_norm = normalize_to_01(physics.setpoint, 0.0, self._process_variable_max)
        self._error: float = 0.0
        self._prev_error: float = 0.0
        self._integral_error: float = 0.0

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([1.0, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

    def _unpack_action_value(self, action: np.ndarray) -> float:
        return float(action[0])

    def _apply_control(self, control_output: int) -> int:
        return self._physics.step(control_output)

    def _compute_error(self, process_variable_norm: float) -> None:
        self._prev_error = self._error
        self._error = self._setpoint_norm - process_variable_norm
        self._integral_error += self._error

    def _build_observation(
        self, process_variable: float, control_output: int
    ) -> np.ndarray:
        process_variable_norm = normalize_to_01(
            process_variable, 0.0, self._process_variable_max
        )
        self._compute_error(process_variable_norm)
        d_error_dt = self._error - self._prev_error
        return np.array(
            [self._error, d_error_dt, self._integral_error],
            dtype=np.float32,
        )

    def _compute_reward(self, observation: np.ndarray, action: np.ndarray) -> float:
        error = float(observation[0])
        return normalize_to_minus1_plus1(-abs(error), -1.0, 0.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step += 1

        control_output_norm = self._unpack_action_value(action)
        control_output = int(round(denormalize_from_minus1_plus1(
            control_output_norm, self._control_min, self._control_max
        )))
        process_variable = self._apply_control(control_output)

        observation = self._build_observation(process_variable, control_output)
        reward = self._compute_reward(observation, action)

        log_line = (
            "step: "
            f"step={self._step} "
            f"process_variable={process_variable} setpoint={self._physics.setpoint} "
            f"error={observation[0]} prev_error={self._prev_error} "
            f"d_error_dt={observation[1]} integral_error={observation[2]} "
            f"control_output_norm={control_output_norm} control_output={control_output} "
            f"reward={reward}"
        )
        self._env_logger.log(log_line)

        terminated = truncated = False
        return observation, reward, terminated, truncated, {}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._step = 0
        self._error = 0.0
        self._prev_error = 0.0
        self._integral_error = 0.0

        process_variable, setpoint, control_output = self._physics.reset()
        self._setpoint_norm = normalize_to_01(setpoint, 0.0, self._process_variable_max)

        observation = self._build_observation(process_variable, control_output)

        log_line = (
            "reset: "
            f"process_variable={process_variable} setpoint={setpoint} "
            f"error={observation[0]} prev_error={self._prev_error} "
            f"d_error_dt={observation[1]} integral_error={observation[2]} "
            f"control_output={control_output}"
        )
        self._env_logger.log(log_line)

        return observation, {}

    def close(self) -> None:
        self._physics.close()
        self._base_logger.close()

    @classmethod
    def from_config(cls, config: Config) -> "NeuralPIDEnv":
        base_logger = AsyncFileLogger(log_dir=config.args.log_dir, log_file=config.args.log_file)
        physics = NeuralControllerPhys(
            port=config.args.port,
            timeout=config.args.timeout,
            baudrate=config.args.baudrate,
            setpoint=config.args.setpoint,
            auto_determine_setpoint=config.args.auto_determine_setpoint,
            setpoint_determination_steps=config.args.setpoint_determination_steps,
            setpoint_determination_max_value=config.args.setpoint_determination_max_value,
            setpoint_determination_factor=config.args.setpoint_determination_factor,
            control_min=config.args.control_min,
            control_max=config.args.control_max,
            reset_value=config.args.reset_value,
            reset_steps=config.args.reset_steps,
            log_connection=config.args.log_connection,
            base_logger=base_logger,
        )
        return cls(
            physics=physics,
            base_logger=base_logger,
            control_min=config.args.control_min,
            control_max=config.args.control_max,
            process_variable_max=config.args.process_variable_max,
        )
