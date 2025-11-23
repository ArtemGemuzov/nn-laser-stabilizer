from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import gymnasium as gym

from nn_laser_stabilizer.plant import Plant
from nn_laser_stabilizer.logger import AsyncFileLogger
from nn_laser_stabilizer.connection import SerialConnection, MockSerialConnection
from nn_laser_stabilizer.pid import ConnectionToPid, TestConnectionToPid, LoggingConnectionToPid


class PidDeltaTuningEnv(gym.Env):  
    metadata = {"render_modes": []}

    def __init__(
        self,
        # Параметры для соединения
        use_mock: bool = False,
        port: str = "COM1",
        timeout: float = 0.1,
        baudrate: int = 115200,
        # Параметры для TestConnectionToPid (если use_mock=True)
        test_optimal_kp: float = 10.0,
        test_optimal_ki: float = 17.5,
        test_setpoint: float = 1200.0,
        test_max_distance: float = 20.0,
        test_noise_std: float = 30.0,
        # Параметры для логирования соединения
        log_connection: bool = False,
        connection_log_dir: Optional[str | Path] = None,
        connection_log_file: str = "connection.log",
        # Параметры для Plant
        setpoint: float = 1200.0,
        warmup_steps: int = 1000,
        block_size: int = 100,
        burn_in_steps: int = 20,
        control_output_min_threshold: float = 200.0,
        control_output_max_threshold: float = 4096.0,
        force_min_value: int = 2000,
        force_max_value: int = 2500,
        default_min: int = 0,
        default_max: int = 4095,
        # Константы для Plant (PID параметры)
        kp_min: float = 2.5,
        kp_max: float = 12.5,
        kp_start: float = 7.5,
        ki_min: float = 0.0,
        ki_max: float = 20.0,
        ki_start: float = 10.0,
        kd_min: float = 0.0,
        kd_max: float = 0.0,
        kd_start: float = 0.0,
        # Константы для PidDeltaTuningEnv
        error_mean_normalization_factor: float = 400.0,
        error_std_normalization_factor: float = 300.0,
        kp_delta_scale: float = 0.01,
        ki_delta_scale: float = 0.01,
        precision_weight: float = 0.4,
        stability_weight: float = 0.4,
        action_weight: float = 0.2,
        # Параметры для AsyncFileLogger
        log_dir: str | Path = ".",
        log_file: str = "env.log",
    ):
        super().__init__()
        
        self._kp_min = kp_min
        self._kp_max = kp_max
        self._ki_min = ki_min
        self._ki_max = ki_max
        self._error_mean_normalization_factor = error_mean_normalization_factor
        self._error_std_normalization_factor = error_std_normalization_factor
        
        kp_range = kp_max - kp_min
        ki_range = ki_max - ki_min
        self._kp_delta_max = kp_range * kp_delta_scale
        self._ki_delta_max = ki_range * ki_delta_scale
        
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
                optimal_kp=test_optimal_kp,
                optimal_ki=test_optimal_ki,
                setpoint=test_setpoint,
                max_distance=test_max_distance,
                noise_std=test_noise_std,
            )
        else:
            connection = SerialConnection(
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
            shape=(2,),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
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
        error_std = np.std(errors)
        
        error_mean_norm = np.clip(
            error_mean / self._error_mean_normalization_factor, -1.0, 1.0
        )
        error_std_norm = np.clip(
            error_std / self._error_std_normalization_factor, 0.0, 1.0
        )
        
        kp_range = self._kp_max - self._kp_min
        ki_range = self._ki_max - self._ki_min
        kp_norm = np.clip(
            (self.plant.kp - self._kp_min) / kp_range * 2.0 - 1.0, -1.0, 1.0
        )
        ki_norm = np.clip(
            (self.plant.ki - self._ki_min) / ki_range * 2.0 - 1.0, -1.0, 1.0
        )
        
        observation = np.array(
            [error_mean_norm, error_std_norm, kp_norm, ki_norm],
            dtype=np.float32
        )
        
        return observation

    def _compute_reward(self, observation: np.ndarray, action: np.ndarray) -> float:
        error_mean_norm, error_std_norm = observation[0], observation[1]
        
        # 1. Штраф за неточность (чем больше ошибка, тем больше штраф)
        precision_penalty = -np.abs(error_mean_norm)      # [-1, 0]
        # 2. Штраф за нестабильность (чем больше разброс, тем больше штраф)
        stability_penalty = -np.abs(error_std_norm)       # [-1, 0]
        # 3. Штраф за действие (чем больше изменение, тем больше штраф)
        action_penalty = -np.mean(np.abs(action))  # [-1, 0]
        
        total_reward = (
            self._precision_weight * precision_penalty + 
            self._stability_weight * stability_penalty + 
            self._action_weight * action_penalty
        )
        
        return 2 * total_reward + 1

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        delta_kp_norm, delta_ki_norm = action[0], action[1]

        delta_kp = delta_kp_norm * self._kp_delta_max
        delta_ki = delta_ki_norm * self._ki_delta_max

        self.plant.update_pid(delta_kp, delta_ki, 0.0)
        process_variables, control_outputs, setpoint, should_reset = self.plant.step()
        self._step += 1

        observation = self._build_observation(
            process_variables, control_outputs, setpoint
        )
        
        action_array = np.array([delta_kp_norm, delta_ki_norm], dtype=np.float32)
        reward = self._compute_reward(observation, action_array)
        
        log_line = (
            f"step={self._step} "
            f"kp={self.plant.kp:.4f} ki={self.plant.ki:.4f} kd={self.plant.kd:.4f} "
            f"delta_kp_norm={delta_kp_norm:.4f} delta_ki_norm={delta_ki_norm:.4f} "
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
        process_variables, control_outputs, setpoint = self.plant.reset()

        observation = self._build_observation(
            process_variables, control_outputs, setpoint
        )
        
        log_line = (
            f"reset "
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