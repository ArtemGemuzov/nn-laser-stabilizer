from typing import Optional
from pathlib import Path

import numpy as np
import gymnasium as gym

from nn_laser_stabilizer.logger import AsyncFileLogger, PrefixedLogger
from nn_laser_stabilizer.envs.neural_controller_phys import NeuralControllerPhys
from nn_laser_stabilizer.time import CallIntervalTracker


class NeuralControllerDeltaEnv(gym.Env):
    LOG_PREFIX = "ENV"

    metadata = {"render_modes": []}

    def __init__(
        self,
        # Параметры для соединения
        port: str,
        timeout: float,
        baudrate: int,
        # Параметры для работы с установкой
        setpoint: int,
        # Параметры автоматического определения setpoint
        auto_determine_setpoint: bool,
        setpoint_determination_steps: int,
        setpoint_determination_max_value: int,
        setpoint_determination_factor: float,
        # Параметры диапазона управления
        control_min: int,
        control_max: int,
        # Максимальное приращение управления за один шаг (в единицах управления)
        max_control_delta: int,
        # Сброс: фиксированное значение и число шагов в начале эпизода
        reset_value: int,
        reset_steps: int,
        # Параметры нормализации наблюдений
        process_variable_max: int,
        # Параметры для логирования окружения
        log_dir: str | Path,
        log_file: str,
    ):
        super().__init__()

        self._control_min = int(control_min)
        self._control_max = int(control_max)
        self._max_control_delta = int(max_control_delta)

        self._process_variable_max = float(process_variable_max)

        self._base_logger = AsyncFileLogger(log_dir=log_dir, log_file=log_file)
        self._env_logger = PrefixedLogger(self._base_logger, NeuralControllerDeltaEnv.LOG_PREFIX)

        self._physics = NeuralControllerPhys(
            port=port,
            timeout=timeout,
            baudrate=baudrate,
            setpoint=setpoint,
            auto_determine_setpoint=auto_determine_setpoint,
            setpoint_determination_steps=setpoint_determination_steps,
            setpoint_determination_max_value=setpoint_determination_max_value,
            setpoint_determination_factor=setpoint_determination_factor,
            control_min=control_min,
            control_max=control_max,
            reset_value=reset_value,
            reset_steps=reset_steps,
            base_logger=self._base_logger,
        )

        self._setpoint_norm = float(setpoint) / self._process_variable_max

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
        delta = action * self._max_control_delta
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

    def _normalize_control_output(self, control_output: int) -> float:
        span = float(self._control_max - self._control_min)
        norm_01 = float(control_output - self._control_min) / span
        return 2.0 * norm_01 - 1.0

    def _normalize_process_variable(self, process_variable: int) -> float:
        return float(np.clip(process_variable / self._process_variable_max, 0.0, 1.0))

    def _build_observation(self, control_output_norm: float) -> np.ndarray:
        return np.array(
            [self._error, control_output_norm],
            dtype=np.float32,
        )

    def _compute_reward(self) -> float:
        return 1.0 - 2.0 * abs(self._error)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        step_interval = self._step_interval_tracker.tick()

        action_value = float(action[0])
        delta = self._map_action_to_delta(action_value)
        new_control = self._map_delta_to_control(delta)
        process_variable = self._physics.step(new_control)
        self._current_control_output = new_control

        process_variable_norm = self._normalize_process_variable(process_variable)

        self._compute_error(process_variable_norm)
        control_output_norm = self._normalize_control_output(self._current_control_output)
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
        self._setpoint_norm = float(setpoint) / self._process_variable_max
        self._current_control_output = control_output

        process_variable_norm = self._normalize_process_variable(process_variable)
        self._compute_error(process_variable_norm)
        control_output_norm = self._normalize_control_output(
            self._current_control_output
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
