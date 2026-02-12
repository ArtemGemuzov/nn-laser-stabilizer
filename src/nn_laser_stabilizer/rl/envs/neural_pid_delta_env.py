import json
from functools import partial
from typing import Optional

import numpy as np
import gymnasium as gym

from nn_laser_stabilizer.logger import AsyncFileLogger, Logger
from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.envs.base_env import BaseEnv
from nn_laser_stabilizer.rl.envs.bounded_value import BoundedValue
from nn_laser_stabilizer.rl.envs.plant_backend import ExperimentalPlantBackend, PlantBackend
from nn_laser_stabilizer.normalize import (
    denormalize_from_minus1_plus1,
    normalize_to_01,
)
from nn_laser_stabilizer.time import CallIntervalTracker


class NeuralPIDDeltaEnv(BaseEnv):
    LOG_SOURCE = "env"

    def __init__(
        self,
        *,
        backend: PlantBackend,
        base_logger: Logger,
        control_min: int,
        control_max: int,
        max_control_delta: int,
        process_variable_max: int,
        reset_value: int,
        reset_steps: int,
    ):
        super().__init__()

        self._normalize_pv = partial(
            normalize_to_01,
            min_val=0.0,
            max_val=float(process_variable_max),
        )
        self._denormalize_delta = partial(
            denormalize_from_minus1_plus1,
            min_val=-float(max_control_delta),
            max_val=float(max_control_delta),
        )

        self._logger = base_logger
        self._backend = backend

        self._step_interval_tracker = CallIntervalTracker(time_multiplier=1e6)
        self._step: int = 0

        self._setpoint_norm = self._normalize_pv(backend.setpoint)
        self._reset_value = reset_value
        self._reset_steps = reset_steps
        self._current_control_output = BoundedValue(control_min, control_max, 0)
        self._error_prev: float = 0.0
        self._raw_error_prev: float = 0.0
        self._error_prev_prev: float = 0.0
        self._raw_error_prev_prev: float = 0.0

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
    ) -> tuple[np.ndarray, dict]:
        process_variable_norm = self._normalize_pv(float(process_variable))
        error = self._setpoint_norm - process_variable_norm
        raw_error = float(self._backend.setpoint - process_variable)

        observation = np.array(
            [error, self._error_prev, self._error_prev_prev],
            dtype=np.float32,
        )
        info = {
            "env.cur_error": raw_error,
            "env.prev_error": self._raw_error_prev,
            "env.prev_prev_error": self._raw_error_prev_prev,
        }

        self._error_prev_prev = self._error_prev
        self._raw_error_prev_prev = self._raw_error_prev
        self._error_prev = error
        self._raw_error_prev = raw_error

        return observation, info

    def _compute_reward(self, observation: np.ndarray, action: np.ndarray) -> float:
        error = float(observation[0])
        return -abs(error)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        step_interval = self._step_interval_tracker.tick()
        self._step += 1

        delta_norm = self._unpack_action_value(action)
        delta = self._denormalize_delta(delta_norm)
        control_output = self._update_control_output(delta)
        process_variable = self._apply_control(control_output)

        observation, info = self._build_observation(process_variable, control_output)
        reward = self._compute_reward(observation, action)

        self._logger.log(json.dumps({
            "source": self.LOG_SOURCE,
            "event": "step",
            "step": self._step,
            "process_variable": process_variable,
            "setpoint": self._backend.setpoint,
            "error": float(observation[0]),
            "error_prev": float(observation[1]),
            "error_prev_prev": float(observation[2]),
            "delta_norm": delta_norm,
            "delta": delta,
            "control_output": control_output,
            "reward": reward,
            "step_interval_us": step_interval,
        }))

        terminated = truncated = False
        return observation, reward, terminated, truncated, info

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
        self._raw_error_prev = 0.0
        self._raw_error_prev_prev = 0.0

        self._backend.reset()
        self._setpoint_norm = self._normalize_pv(self._backend.setpoint)
        self._current_control_output.value = self._reset_value

        info = {}
        for _ in range(self._reset_steps):
            process_variable = self._apply_control(self._reset_value)
            observation, info = self._build_observation(process_variable, self._reset_value)

        self._logger.log(json.dumps({
            "source": self.LOG_SOURCE,
            "event": "reset",
            "setpoint": self._backend.setpoint,
        }))

        return observation, info

    def close(self) -> None:
        self._backend.close()
        self._logger.close()

    @classmethod
    def from_config(cls, config: Config) -> "NeuralPIDDeltaEnv":
        logger = AsyncFileLogger(
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
            log_connection=config.args.log_connection,
            base_logger=logger,
        )
        return cls(
            backend=backend,
            base_logger=logger,
            control_min=config.args.control_min,
            control_max=config.args.control_max,
            max_control_delta=config.args.max_control_delta,
            process_variable_max=config.args.process_variable_max,
            reset_value=config.args.reset_value,
            reset_steps=config.args.reset_steps,
        )
