from typing import Optional
from pathlib import Path

import numpy as np
import gymnasium as gym

from nn_laser_stabilizer.logger import AsyncFileLogger, PrefixedLogger
from nn_laser_stabilizer.envs.neural_pid_phys import NeuralPIDPhysics


class NeuralPIDEnv(gym.Env):
    LOG_PREFIX = "ENV"

    metadata = {"render_modes": []}

    def __init__(
        self,
        # Параметры для соединения
        port: str,
        timeout: float,
        baudrate: int,
        # Параметры для логирования соединения
        log_connection: bool,
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
        # Параметры нормализации наблюдений
        process_variable_max: int,
        # Параметры для логирования окружения
        log_dir: str | Path,
        log_file: str,
    ):
        super().__init__()

        self._control_min = int(control_min)
        self._control_max = int(control_max)
        
        self._process_variable_max = float(process_variable_max)

        self._base_logger = AsyncFileLogger(log_dir=log_dir, log_file=log_file)
        self._env_logger = PrefixedLogger(self._base_logger, NeuralPIDEnv.LOG_PREFIX)

        self._physics = NeuralPIDPhysics(
            port=port,
            timeout=timeout,
            baudrate=baudrate,
            log_connection=log_connection,
            setpoint=setpoint,
            auto_determine_setpoint=auto_determine_setpoint,
            setpoint_determination_steps=setpoint_determination_steps,
            setpoint_determination_max_value=setpoint_determination_max_value,
            setpoint_determination_factor=setpoint_determination_factor,
            control_min=control_min,
            control_max=control_max,
            base_logger=self._base_logger,
        )

        self._setpoint_norm = float(setpoint) / self._process_variable_max

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
        norm = (action + 1.0) / 2.0
        control = self._control_min + norm * (self._control_max - self._control_min)
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
        return float(-abs(self._error))

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

        self._physics.open_and_warmup()

        process_variable, neutral_control_output = self._physics.neutral_measure()
      
        process_variable_norm = float(process_variable) / self._process_variable_max
        self._compute_error(process_variable_norm)
        observation = self._build_observation()

        d_error_dt = self._error - self._prev_error

        log_line = (
            "reset: "
            f"process_variable={process_variable} setpoint={self._physics.setpoint} "
            f"error={self._error} prev_error={self._prev_error} "
            f"d_error_dt={d_error_dt} integral_error={self._integral_error} "
            f"neutral_control_output={neutral_control_output}"
        )
        self._env_logger.log(log_line)

        return observation, {}

    def close(self) -> None:
        self._physics.close()
        self._base_logger.close()