from typing import Optional
import time

import numpy as np
import gymnasium as gym

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.envs.base_env import BaseEnv
from nn_laser_stabilizer.envs.pid_delta_tuning_phys import PidDeltaTuningPhys
from nn_laser_stabilizer.logger import AsyncFileLogger, Logger, PrefixedLogger
from nn_laser_stabilizer.connection.pid_protocol import PidProtocol


class PidDeltaTuningEnv(BaseEnv):
    LOG_PREFIX = "ENV"

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        phys: PidDeltaTuningPhys,
        base_logger: Logger,
        kp_min: float,
        kp_max: float,
        kp_delta_scale: float,
        ki_min: float,
        ki_max: float,
        ki_delta_scale: float,
        kd_min: float,
        kd_max: float,
        kd_delta_scale: float,
        error_mean_normalization_factor: float,
        error_std_normalization_factor: float,
        precision_weight: float,
        stability_weight: float,
        action_weight: float,
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

        self._base_logger = base_logger
        self._env_logger = PrefixedLogger(self._base_logger, PidDeltaTuningEnv.LOG_PREFIX)
        self.phys = phys

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
            (self.phys.kp - self._kp_min) / self._kp_range * 2.0 - 1.0, -1.0, 1.0
        )
        ki_norm = np.clip(
            (self.phys.ki - self._ki_min) / self._ki_range * 2.0 - 1.0, -1.0, 1.0
        )
        kd_norm = np.clip(
            (self.phys.kd - self._kd_min) / self._kd_range * 2.0 - 1.0, -1.0, 1.0
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

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        delta_kp_norm, delta_ki_norm, delta_kd_norm = action[0], action[1], action[2]

        delta_kp = delta_kp_norm * self._kp_delta_max
        delta_ki = delta_ki_norm * self._ki_delta_max
        delta_kd = delta_kd_norm * self._kd_delta_max

        self.phys.update_pid(delta_kp, delta_ki, delta_kd)
        process_variables, control_outputs, setpoint, should_reset = self.phys.step()
        self._step += 1

        observation = self._build_observation(
            process_variables, control_outputs, setpoint
        )
        
        action_array = np.array([delta_kp_norm, delta_ki_norm, delta_kd_norm], dtype=np.float32)
        reward = self._compute_reward(observation, action_array)
        
        log_line = (
            f"step: step={self._step} time={time.time()} "
            f"kp={self.phys.kp:.{PidProtocol.KP_DECIMAL_PLACES}f} "
            f"ki={self.phys.ki:.{PidProtocol.KI_DECIMAL_PLACES}f} "
            f"kd={self.phys.kd:.{PidProtocol.KD_DECIMAL_PLACES}f} "
            f"delta_kp_norm={action[0]} delta_ki_norm={action[1]} delta_kd_norm={action[2]} "
            f"error_mean_norm={observation[0]} error_std_norm={observation[1]} "
            f"reward={reward} should_reset={should_reset}"
        )
        self._env_logger.log(log_line)   

        terminated = should_reset
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        process_variables, control_outputs, setpoint, _ = self.phys.reset()

        observation = self._build_observation(
            process_variables, control_outputs, setpoint
        )
        
        log_line = (
            f"reset: time={time.time()} "
            f"kp={self.phys.kp:.{PidProtocol.KP_DECIMAL_PLACES}f} "
            f"ki={self.phys.ki:.{PidProtocol.KI_DECIMAL_PLACES}f} "
            f"kd={self.phys.kd:.{PidProtocol.KD_DECIMAL_PLACES}f} "
            f"error_mean_norm={observation[0]} error_std_norm={observation[1]} "
        )
        self._env_logger.log(log_line) 

        info = {}
        return observation, info

    def close(self) -> None:
        self.phys.close()
        self._base_logger.close()

    @classmethod
    def from_config(cls, config: Config) -> "PidDeltaTuningEnv":
        base_logger = AsyncFileLogger(log_dir=config.args.log_dir, log_file=config.args.log_file)
        phys = PidDeltaTuningPhys(
            logger=base_logger,
            port=config.args.port,
            timeout=config.args.timeout,
            baudrate=config.args.baudrate,
            log_connection=config.args.log_connection,
            setpoint=config.args.setpoint,
            warmup_steps=config.args.warmup_steps,
            block_size=config.args.block_size,
            burn_in_steps=config.args.burn_in_steps,
            control_output_min_threshold=config.args.control_output_min_threshold,
            control_output_max_threshold=config.args.control_output_max_threshold,
            force_min_value=config.args.force_min_value,
            force_max_value=config.args.force_max_value,
            default_min=config.args.default_min,
            default_max=config.args.default_max,
            kp_min=config.args.kp_min,
            kp_max=config.args.kp_max,
            kp_start=config.args.kp_start,
            ki_min=config.args.ki_min,
            ki_max=config.args.ki_max,
            ki_start=config.args.ki_start,
            kd_min=config.args.kd_min,
            kd_max=config.args.kd_max,
            kd_start=config.args.kd_start,
            auto_determine_setpoint=config.args.auto_determine_setpoint,
            setpoint_determination_steps=config.args.setpoint_determination_steps,
            setpoint_determination_max_value=config.args.setpoint_determination_max_value,
            setpoint_determination_factor=config.args.setpoint_determination_factor,
        )
        return cls(
            phys=phys,
            base_logger=base_logger,
            kp_min=config.args.kp_min,
            kp_max=config.args.kp_max,
            kp_delta_scale=config.args.kp_delta_scale,
            ki_min=config.args.ki_min,
            ki_max=config.args.ki_max,
            ki_delta_scale=config.args.ki_delta_scale,
            kd_min=config.args.kd_min,
            kd_max=config.args.kd_max,
            kd_delta_scale=config.args.kd_delta_scale,
            error_mean_normalization_factor=config.args.error_mean_normalization_factor,
            error_std_normalization_factor=config.args.error_std_normalization_factor,
            precision_weight=config.args.precision_weight,
            stability_weight=config.args.stability_weight,
            action_weight=config.args.action_weight,
        )