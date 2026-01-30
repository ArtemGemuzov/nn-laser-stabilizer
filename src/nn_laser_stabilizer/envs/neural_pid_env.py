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

    metadata = {"render_modes": []}

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

        self._setpoint_norm = normalize_to_01(physics.setpoint, 0.0, self._process_variable_max)

        self._error: float = 0.0
        self._prev_error: float = 0.0
        self._integral_error: float = 0.0
        self._step: int = 0

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

    def _map_action_to_control(self, action: float) -> int:
        """Линейное отображение действия из [-1, 1] в [control_min, control_max]."""
        control = denormalize_from_minus1_plus1(
            action, self._control_min, self._control_max
        )
        return int(round(control))

    def _compute_error(self, process_variable_norm: float) -> None:
        self._prev_error = self._error
        self._error = self._setpoint_norm - process_variable_norm
        self._integral_error += self._error

    def _build_observation(self) -> np.ndarray:
        d_error_dt = self._error - self._prev_error
        
        return np.array(
            [self._error, d_error_dt, self._integral_error],
            dtype=np.float32,
        )

    def _compute_reward(self) -> float:
        return normalize_to_minus1_plus1(-abs(self._error), -1.0, 0.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_value = float(action[0])
        control_output = self._map_action_to_control(action_value)
        process_variable = self._physics.step(control_output)
        process_variable_norm = float(process_variable) / self._process_variable_max

        self._compute_error(process_variable_norm)
        obs = self._build_observation()
        reward = self._compute_reward()

        self._step += 1

        d_error_dt = self._error - self._prev_error

        log_line = (
            "step: "
            f"step={self._step} "
            f"process_variable={process_variable} setpoint={self._physics.setpoint} "
            f"error={self._error} prev_error={self._prev_error} "
            f"d_error_dt={d_error_dt} integral_error={self._integral_error} "
            f"action={action_value} control_output={control_output} "
            f"reward={reward}"
        )
        self._env_logger.log(log_line)

        terminated = truncated = False
        return obs, reward, terminated, truncated, {}

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

        process_variable_norm = float(process_variable) / self._process_variable_max
        self._compute_error(process_variable_norm)
        observation = self._build_observation()

        d_error_dt = self._error - self._prev_error

        log_line = (
            "reset: "
            f"process_variable={process_variable} setpoint={setpoint} "
            f"error={self._error} prev_error={self._prev_error} "
            f"d_error_dt={d_error_dt} integral_error={self._integral_error} "
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
