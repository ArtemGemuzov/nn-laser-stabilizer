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


class NeuralPIDDeltaEnv(BaseEnv):
    LOG_PREFIX = "ENV"

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
        self._env_logger = PrefixedLogger(
            self._base_logger, NeuralPIDDeltaEnv.LOG_PREFIX
        )
        self._physics = physics

        self._step_interval_tracker = CallIntervalTracker(time_multiplier=1e6)
        self._step: int = 0

        self._setpoint_norm = normalize_to_01(
            physics.setpoint, 0.0, self._process_variable_max
        )
        self._current_control_output: int = 0
        self._error_prev: float = 0.0
        self._error_prev_prev: float = 0.0

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _unpack_action_value(self, action: np.ndarray) -> float:
        return float(action[0])

    def _update_control_output(self, delta: float) -> int:
        delta_int = int(round(delta))
        new_control = int(
            np.clip(
                self._current_control_output + delta_int,
                self._control_min,
                self._control_max,
            )
        )
        self._current_control_output = new_control
        return new_control

    def _apply_control(self, control_output: int) -> int:
        return self._physics.step(control_output)

    def _build_observation(
        self, process_variable: float, control_output: int
    ) -> np.ndarray:
        process_variable_norm = normalize_to_01(
            float(process_variable), 0.0, self._process_variable_max
        )
        error = self._setpoint_norm - process_variable_norm
        control_output_norm = normalize_to_minus1_plus1(
            float(control_output), self._control_min, self._control_max
        )
        observation = np.array(
            [
                control_output_norm,
                error,
                self._error_prev,
                self._error_prev_prev,
            ],
            dtype=np.float32,
        )
        self._error_prev_prev = self._error_prev
        self._error_prev = error
        return observation

    def _compute_reward(self, observation: np.ndarray, action: np.ndarray) -> float:
        error = float(observation[1])
        return normalize_to_minus1_plus1(-abs(error), -1.0, 0.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        step_interval = self._step_interval_tracker.tick()
        self._step += 1

        delta_norm = self._unpack_action_value(action)
        delta = denormalize_from_minus1_plus1(
            delta_norm, -self._max_control_delta, self._max_control_delta
        )
        control_output = self._update_control_output(delta)
        process_variable = self._apply_control(control_output)

        observation = self._build_observation(process_variable, control_output)
        reward = self._compute_reward(observation, action)

        log_line = (
            "step: "
            f"step={self._step} "
            f"process_variable={process_variable} setpoint={self._physics.setpoint} "
            f"error={observation[1]} error_prev={observation[2]} error_prev_prev={observation[3]} "
            f"delta_norm={delta_norm} delta={delta} control_output={control_output} "
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
        self._step_interval_tracker.reset()
        self._error_prev = 0.0
        self._error_prev_prev = 0.0

        process_variable, setpoint, control_output = self._physics.reset()
        self._setpoint_norm = normalize_to_01(
            setpoint, 0.0, self._process_variable_max
        )
        self._current_control_output = control_output

        observation = self._build_observation(process_variable, control_output)

        log_line = (
            "reset: "
            f"process_variable={process_variable} setpoint={setpoint} "
            f"error={observation[1]} control_output={control_output}"
        )
        self._env_logger.log(log_line)

        return observation, {}

    def close(self) -> None:
        self._physics.close()
        self._base_logger.close()

    @classmethod
    def from_config(cls, config: Config) -> "NeuralPIDDeltaEnv":
        base_logger = AsyncFileLogger(
            log_dir=config.args.log_dir, log_file=config.args.log_file
        )
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
