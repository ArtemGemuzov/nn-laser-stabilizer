from functools import partial
from typing import Optional

import numpy as np
import gymnasium as gym

from nn_laser_stabilizer.logger import AsyncFileLogger, Logger, PrefixedLogger
from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.envs.base_env import BaseEnv
from nn_laser_stabilizer.envs.bounded_value import BoundedValue
from nn_laser_stabilizer.envs.plant_backend import ExperimentalPlantBackend, PlantBackend
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
        backend: PlantBackend,
        base_logger: Logger,
        control_min: int,
        control_max: int,
        max_control_delta: int,
        process_variable_max: int,
    ):
        super().__init__()

        self._normalize_pv = partial(normalize_to_01, 0.0, float(process_variable_max))
        self._denormalize_delta = partial(
            denormalize_from_minus1_plus1,
            -float(max_control_delta),
            float(max_control_delta),
        )
        self._normalize_reward = partial(normalize_to_minus1_plus1, -1.0, 0.0)

        self._base_logger = base_logger
        self._env_logger = PrefixedLogger(
            self._base_logger, NeuralPIDDeltaEnv.LOG_PREFIX
        )
        self._backend = backend

        self._step_interval_tracker = CallIntervalTracker(time_multiplier=1e6)
        self._step: int = 0

        self._setpoint_norm = self._normalize_pv(backend.setpoint)
        self._current_control_output = BoundedValue(control_min, control_max, 0)
        self._error_prev: float = 0.0
        self._error_prev_prev: float = 0.0

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _unpack_action_value(self, action: np.ndarray) -> float:
        return float(action[0])

    def _update_control_output(self, delta: float) -> int:
        return self._current_control_output.add(int(round(delta)))

    def _apply_control(self, control_output: int) -> int:
        return self._backend.exchange(control_output)

    def _build_observation(
        self, process_variable: float, control_output: int
    ) -> np.ndarray:
        process_variable_norm = self._normalize_pv(float(process_variable))
        error = self._setpoint_norm - process_variable_norm
        observation = np.array(
            [error, self._error_prev, self._error_prev_prev],
            dtype=np.float32,
        )
        self._error_prev_prev = self._error_prev
        self._error_prev = error
        return observation

    def _compute_reward(self, observation: np.ndarray, action: np.ndarray) -> float:
        error = float(observation[0])
        return self._normalize_reward(-abs(error))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        step_interval = self._step_interval_tracker.tick()
        self._step += 1

        delta_norm = self._unpack_action_value(action)
        delta = self._denormalize_delta(delta_norm)
        control_output = self._update_control_output(delta)
        process_variable = self._apply_control(control_output)

        observation = self._build_observation(process_variable, control_output)
        reward = self._compute_reward(observation, action)

        log_line = (
            "step: "
            f"step={self._step} "
            f"process_variable={process_variable} setpoint={self._backend.setpoint} "
            f"error={observation[0]} error_prev={observation[1]} error_prev_prev={observation[2]} "
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

        process_variable, setpoint, control_output = self._backend.reset()
        self._setpoint_norm = self._normalize_pv(setpoint)
        self._current_control_output.value = control_output

        observation = self._build_observation(process_variable, control_output)

        log_line = (
            "reset: "
            f"process_variable={process_variable} setpoint={setpoint} "
            f"error={observation[0]} control_output={control_output}"
        )
        self._env_logger.log(log_line)

        return observation, {}

    def close(self) -> None:
        self._backend.close()
        self._base_logger.close()

    @classmethod
    def from_config(cls, config: Config) -> "NeuralPIDDeltaEnv":
        base_logger = AsyncFileLogger(
            log_dir=config.args.log_dir, log_file=config.args.log_file
        )
        backend = ExperimentalPlantBackend(
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
            backend=backend,
            base_logger=base_logger,
            control_min=config.args.control_min,
            control_max=config.args.control_max,
            max_control_delta=config.args.max_control_delta,
            process_variable_max=config.args.process_variable_max,
        )
