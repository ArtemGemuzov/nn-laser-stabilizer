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
from nn_laser_stabilizer.time import CallIntervalTracker


class NeuralControllerDeltaEnv(BaseEnv):
    LOG_PREFIX = "ENV"

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        physics: NeuralControllerPhys,
        base_logger: Logger,
        control_min: int,
        control_max: int,
        max_control_delta: int,
        process_variable_max: float,
    ):
        super().__init__()

        self._control_min = int(control_min)
        self._control_max = int(control_max)
        self._max_control_delta = int(max_control_delta)
        self._process_variable_max = float(process_variable_max)

        self._base_logger = base_logger
        self._env_logger = PrefixedLogger(self._base_logger, NeuralControllerDeltaEnv.LOG_PREFIX)
        self._physics = physics

        self._setpoint_norm = normalize_to_01(physics.setpoint, 0.0, self._process_variable_max)

        self._error: float = 0.0
        self._step: int = 0
        self._current_control_output: int = 0

        self._step_interval_tracker = CallIntervalTracker(time_multiplier=1e6)

        # Действие: приращение в [-1, 1], интерпретируется как доля от max_control_delta
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _map_action_to_delta(self, action: float) -> int:
        delta = denormalize_from_minus1_plus1(
            action, -self._max_control_delta, self._max_control_delta
        )
        return int(round(delta))

    def _map_delta_to_control(self, delta: int) -> int:
        new_control = np.clip(
            self._current_control_output + delta,
            self._control_min,
            self._control_max,
        )
        return int(new_control)

    def _compute_error(self, process_variable_norm: float) -> None:
        self._error = self._setpoint_norm - process_variable_norm

    def _build_observation(self, control_output_norm: float) -> np.ndarray:
        return np.array(
            [self._error, control_output_norm],
            dtype=np.float32,
        )

    def _compute_reward(self) -> float:
        return normalize_to_minus1_plus1(-abs(self._error), -1.0, 0.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        step_interval = self._step_interval_tracker.tick()

        action_value = float(action[0])
        delta = self._map_action_to_delta(action_value)
        new_control = self._map_delta_to_control(delta)
        process_variable = self._physics.step(new_control)
        self._current_control_output = new_control

        process_variable_norm = normalize_to_01(process_variable, 0.0, self._process_variable_max)

        self._compute_error(process_variable_norm)
        control_output_norm = normalize_to_minus1_plus1(
            self._current_control_output, self._control_min, self._control_max
        )
        observation = self._build_observation(control_output_norm)
        reward = self._compute_reward()

        self._step += 1

        log_line = (
            "step: "
            f"step={self._step} "
            f"process_variable={process_variable} setpoint={self._physics.setpoint} error={self._error} "
            f"action={action_value} delta={delta} control_output={new_control} "
            f"reward={reward} "
            f"step_interval={step_interval}us"
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
        self._step_interval_tracker.reset()

        process_variable, setpoint, control_output = self._physics.reset()
        self._setpoint_norm = normalize_to_01(setpoint, 0.0, self._process_variable_max)
        self._current_control_output = control_output

        process_variable_norm = normalize_to_01(process_variable, 0.0, self._process_variable_max)
        self._compute_error(process_variable_norm)
        control_output_norm = normalize_to_minus1_plus1(
            self._current_control_output, self._control_min, self._control_max
        )
        observation = self._build_observation(control_output_norm)

        log_line = (
            "reset: "
            f"process_variable={process_variable} setpoint={setpoint} error={self._error} "
            f"control_output={control_output}"
        )
        self._env_logger.log(log_line)

        return observation, {}

    def close(self) -> None:
        self._physics.close()
        self._base_logger.close()

    @classmethod
    def from_config(cls, config: Config) -> "NeuralControllerDeltaEnv":
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
            max_control_delta=config.args.max_control_delta,
            process_variable_max=config.args.process_variable_max,
        )
