from typing import Optional, Tuple
from pathlib import Path
import time

import numpy as np
import gymnasium as gym

from nn_laser_stabilizer.plant import Plant
from nn_laser_stabilizer.logger import AsyncFileLogger
from nn_laser_stabilizer.connection import COMConnection, MockSerialConnection
from nn_laser_stabilizer.pid import ConnectionToPid, TestConnectionToPid, LoggingConnectionToPid


class PidDeltaTuningEnv(gym.Env):  
    metadata = {"render_modes": []}

    def __init__(
        self,
        # Параметры для соединения
        use_mock: bool,
        port: str,
        timeout: float,
        baudrate: int,
        # Параметры для логирования соединения
        log_connection: bool,
        connection_log_dir: Optional[str | Path],
        connection_log_file: str,
        # Параметры для Plant
        setpoint: float,
        warmup_steps: int,
        block_size: int,
        burn_in_steps: int,
        control_output_min_threshold: float,
        control_output_max_threshold: float,
        force_min_value: int,
        force_max_value: int,
        default_min: int,
        default_max: int,
        # Константы для Plant (PID параметры)
        kp_min: float,
        kp_max: float,
        kp_start: float,
        kp_delta_scale: float,
        ki_min: float,
        ki_max: float,
        ki_start: float,
        ki_delta_scale: float,
        kd_min: float,
        kd_max: float,
        kd_start: float,
        kd_delta_scale: float,
        # Константы для PidDeltaTuningEnv
        error_mean_normalization_factor: float,
        error_std_normalization_factor: float,
        precision_weight: float,
        stability_weight: float,
        action_weight: float,
        # Параметры для AsyncFileLogger
        log_dir: str | Path,
        log_file: str,
    ):
        super().__init__()

        self._error_mean_normalization_factor = error_mean_normalization_factor
        self._error_std_normalization_factor = error_std_normalization_factor
        
        self._kp_min = kp_min
        self._kp_max = kp_max
        self._kp_range = kp_max - kp_min
        self._kp_delta_max = self._kp_range * kp_delta_scale

        self._ki_min = ki_min
        self._ki_max = ki_max
        self._ki_range = ki_max - ki_min
        self._ki_delta_max = self._ki_range * ki_delta_scale

        self._kd_min = kd_min
        self._kd_max = kd_max
        self._kd_range = kd_max - kd_min
        self._kd_delta_max = self._kd_range * kd_delta_scale
        
        self._precision_weight = precision_weight
        self._stability_weight = stability_weight
        self._action_weight = action_weight
        
        if use_mock:
            connection = MockSerialConnection(
                port=port,
                timeout=timeout,
                baudrate=baudrate
            )
            pid_connection = TestConnectionToPid(
                connection=connection,
                kp_min=kp_min,
                kp_max=kp_max,
                ki_min=ki_min,
                ki_max=ki_max,
                kd_min=kd_min,
                kd_max=kd_max,
                setpoint=setpoint,
            )
        else:
            connection = COMConnection(
                port=port,
                timeout=timeout,
                baudrate=baudrate,
            )
            pid_connection = ConnectionToPid(connection=connection)
        
        self._connection_logger: Optional[AsyncFileLogger] = None
        if log_connection:
            if connection_log_dir is None:
                connection_log_dir = log_dir
            self._connection_logger = AsyncFileLogger(log_dir=connection_log_dir, log_file=connection_log_file)
            pid_connection = LoggingConnectionToPid(connection_to_pid=pid_connection, logger=self._connection_logger)
        
        self.plant = Plant(
            pid_connection=pid_connection,
            setpoint=setpoint,
            warmup_steps=warmup_steps,
            block_size=block_size,
            burn_in_steps=burn_in_steps,
            control_output_min_threshold=control_output_min_threshold,
            control_output_max_threshold=control_output_max_threshold,
            force_min_value=force_min_value,
            force_max_value=force_max_value,
            default_min=default_min,
            default_max=default_max,
            kp_min=kp_min,
            kp_max=kp_max,
            kp_start=kp_start,
            ki_min=ki_min,
            ki_max=ki_max,
            ki_start=ki_start,
            kd_min=kd_min,
            kd_max=kd_max,
            kd_start=kd_start,
        )
        
        self.logger = AsyncFileLogger(log_dir=log_dir, log_file=log_file)
      
        self._step = 0
        
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def _build_observation(
        self,
        process_variables: np.ndarray,
        control_outputs: np.ndarray,
        setpoint: float,
    ) -> np.ndarray:
        errors = process_variables - setpoint
        
        error_mean = np.mean(errors)
        error_mean_norm = np.clip(
            error_mean / self._error_mean_normalization_factor, -1.0, 1.0
        )

        error_std = np.std(errors)
        error_std_norm = np.clip(
            error_std / self._error_std_normalization_factor, 0.0, 1.0
        )
        
        kp_norm = np.clip(
            (self.plant.kp - self._kp_min) / self._kp_range * 2.0 - 1.0, -1.0, 1.0
        )
        ki_norm = np.clip(
            (self.plant.ki - self._ki_min) / self._ki_range * 2.0 - 1.0, -1.0, 1.0
        )
        kd_norm = np.clip(
            (self.plant.kd - self._kd_min) / self._kd_range * 2.0 - 1.0, -1.0, 1.0
        )
        
        observation = np.array(
            [error_mean_norm, error_std_norm, kp_norm, ki_norm, kd_norm],
            dtype=np.float32
        )
        
        return observation

    def _compute_reward(self, observation: np.ndarray, action: np.ndarray) -> float:
        error_mean_norm, error_std_norm = observation[0], observation[1]
        
        # 1. Штраф за неточность (чем больше ошибка, тем больше штраф)
        precision_penalty = -np.abs(error_mean_norm)      # [-1, 0]
        # 2. Штраф за нестабильность (чем больше разброс, тем больше штраф)
        stability_penalty = -np.abs(error_std_norm)       # [-1, 0]
        # 3. Штраф за действие (чем больше изменения kp/ki/kd, тем больше штраф)
        action_penalty = -np.mean(np.abs(action))  # [-1, 0]
        
        total_reward = (
            self._precision_weight * precision_penalty + 
            self._stability_weight * stability_penalty + 
            self._action_weight * action_penalty
        )
        
        return 2 * total_reward + 1

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        delta_kp_norm, delta_ki_norm, delta_kd_norm = action[0], action[1], action[2]

        delta_kp = delta_kp_norm * self._kp_delta_max
        delta_ki = delta_ki_norm * self._ki_delta_max
        delta_kd = delta_kd_norm * self._kd_delta_max

        self.plant.update_pid(delta_kp, delta_ki, delta_kd)
        process_variables, control_outputs, setpoint, should_reset = self.plant.step()
        self._step += 1

        observation = self._build_observation(
            process_variables, control_outputs, setpoint
        )
        
        action_array = np.array([delta_kp_norm, delta_ki_norm, delta_kd_norm], dtype=np.float32)
        reward = self._compute_reward(observation, action_array)
        
        log_line = (
            f"step={self._step} time={time.time():.6f} "
            f"kp={self.plant.kp:.4f} ki={self.plant.ki:.4f} kd={self.plant.kd:.4f} "
            f"delta_kp_norm={action[0]:.4f} delta_ki_norm={action[1]:.4f} delta_kd_norm={action[2]:.4f} "
            f"error_mean_norm={observation[0]:.4f} error_std_norm={observation[1]:.4f} "
            f"reward={reward:.6f} should_reset={should_reset}"
        )
        self.logger.log(log_line)   

        terminated = should_reset
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        process_variables, control_outputs, setpoint, _ = self.plant.reset()

        observation = self._build_observation(
            process_variables, control_outputs, setpoint
        )
        
        log_line = (
            f"reset time={time.time():.6f} "
            f"kp={self.plant.kp:.4f} ki={self.plant.ki:.4f} kd={self.plant.kd:.4f} "
            f"error_mean_norm={observation[0]:.4f} error_std_norm={observation[1]:.4f} "
        )
        self.logger.log(log_line) 

        info = {}
        return observation, info

    def close(self) -> None:
        self.plant.close()
        self.logger.close()
        if self._connection_logger is not None:
            self._connection_logger.close()


class PendulumNoVelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        
        self.env = gym.make("Pendulum-v1")
    
        self.action_space = self.env.action_space
        
        low = np.delete(self.env.observation_space.low, 2)
        high = np.delete(self.env.observation_space.high, 2)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        filtered_obs = observation[:2]
        return filtered_obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        observation, info = self.env.reset(seed=seed, options=options)
        filtered_obs = observation[:2]
        return filtered_obs, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


_CUSTOM_ENV_MAP: dict[str, type] = {
    "PendulumNoVelEnv": PendulumNoVelEnv,
    "PidDeltaTuningEnv": PidDeltaTuningEnv
}