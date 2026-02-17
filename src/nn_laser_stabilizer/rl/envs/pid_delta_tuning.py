from functools import partial
from typing import Optional
import time

import numpy as np
import gymnasium as gym

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.envs.base_env import BaseEnv
from nn_laser_stabilizer.rl.envs.bounded_value import BoundedValue
from nn_laser_stabilizer.utils.normalize import (
    denormalize_from_minus1_plus1,
    normalize_to_minus1_plus1,
)
from nn_laser_stabilizer.rl.envs.pid_loop_backend import ExperimentalPidLoopBackend, PidLoopBackend
from nn_laser_stabilizer.utils.logger import AsyncFileLogger, Logger, PrefixedLogger
from nn_laser_stabilizer.connection.pid_protocol import PidProtocol


class PidDeltaTuning(BaseEnv):
    LOG_PREFIX = "ENV"

    def __init__(
        self,
        *,
        backend: PidLoopBackend,
        base_logger: Logger,
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
        error_mean_normalization_factor: float,
        error_std_normalization_factor: float,
        precision_weight: float,
        stability_weight: float,
        action_weight: float,
    ):
        super().__init__()

        self._error_mean_normalization_factor = error_mean_normalization_factor
        self._error_std_normalization_factor = error_std_normalization_factor

        kp_delta_max = (kp_max - kp_min) * kp_delta_scale
        ki_delta_max = (ki_max - ki_min) * ki_delta_scale
        kd_delta_max = (kd_max - kd_min) * kd_delta_scale
        self._normalize_kp = partial(
            normalize_to_minus1_plus1,
            min_val=kp_min,
            max_val=kp_max,
        )
        self._normalize_ki = partial(
            normalize_to_minus1_plus1,
            min_val=ki_min,
            max_val=ki_max,
        )
        self._normalize_kd = partial(
            normalize_to_minus1_plus1,
            min_val=kd_min,
            max_val=kd_max,
        )
        self._denormalize_kp_delta = partial(
            denormalize_from_minus1_plus1,
            min_val=-kp_delta_max,
            max_val=kp_delta_max,
        )
        self._denormalize_ki_delta = partial(
            denormalize_from_minus1_plus1,
            min_val=-ki_delta_max,
            max_val=ki_delta_max,
        )
        self._denormalize_kd_delta = partial(
            denormalize_from_minus1_plus1,
            min_val=-kd_delta_max,
            max_val=kd_delta_max,
        )

        self._precision_weight = precision_weight
        self._stability_weight = stability_weight
        self._action_weight = action_weight

        self._base_logger = base_logger
        self._env_logger = PrefixedLogger(self._base_logger, PidDeltaTuning.LOG_PREFIX)
        self._backend = backend

        self._kp = BoundedValue[float](kp_min, kp_max, kp_start)
        self._ki = BoundedValue[float](ki_min, ki_max, ki_start)
        self._kd = BoundedValue[float](kd_min, kd_max, kd_start)
        self._kp_start = kp_start
        self._ki_start = ki_start
        self._kd_start = kd_start

        self._step: int = 0

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
        
        kp_norm = self._normalize_kp(self._kp.value)
        ki_norm = self._normalize_ki(self._ki.value)
        kd_norm = self._normalize_kd(self._kd.value)
        
        return np.array(
            [error_mean_norm, error_std_norm, kp_norm, ki_norm, kd_norm],
            dtype=np.float32
        )

    def _unpack_action_value(
        self, action: np.ndarray
    ) -> tuple[float, float, float]:
        return float(action[0]), float(action[1]), float(action[2])

    def _update_pid_params(
        self, delta_kp_norm: float, delta_ki_norm: float, delta_kd_norm: float
    ) -> None:
        delta_kp = self._denormalize_kp_delta(delta_kp_norm)
        delta_ki = self._denormalize_ki_delta(delta_ki_norm)
        delta_kd = self._denormalize_kd_delta(delta_kd_norm)
        self._kp.add(delta_kp)
        self._ki.add(delta_ki)
        self._kd.add(delta_kd)
        self._kp.value = round(self._kp.value, PidProtocol.KP_DECIMAL_PLACES)
        self._ki.value = round(self._ki.value, PidProtocol.KI_DECIMAL_PLACES)
        self._kd.value = round(self._kd.value, PidProtocol.KD_DECIMAL_PLACES)

    def _apply_control(self) -> tuple[np.ndarray, np.ndarray, float, bool]:
        return self._backend.run_block(self._kp.value, self._ki.value, self._kd.value)

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
        self._step += 1

        delta_kp_norm, delta_ki_norm, delta_kd_norm = self._unpack_action_value(action)
        self._update_pid_params(delta_kp_norm, delta_ki_norm, delta_kd_norm)
        process_variables, control_outputs, setpoint, should_reset = self._apply_control()

        observation = self._build_observation(
            process_variables, control_outputs, setpoint
        )
        reward = self._compute_reward(observation, action)

        log_line = (
            f"step: step={self._step} time={time.time()} "
            f"kp={self._kp.value:.{PidProtocol.KP_DECIMAL_PLACES}f} "
            f"ki={self._ki.value:.{PidProtocol.KI_DECIMAL_PLACES}f} "
            f"kd={self._kd.value:.{PidProtocol.KD_DECIMAL_PLACES}f} "
            f"delta_kp_norm={delta_kp_norm} delta_ki_norm={delta_ki_norm} delta_kd_norm={delta_kd_norm} "
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
        self._kp.value = self._kp_start
        self._ki.value = self._ki_start
        self._kd.value = self._kd_start
        process_variables, control_outputs, setpoint, _ = self._backend.start(
            self._kp.value, self._ki.value, self._kd.value
        )

        observation = self._build_observation(
            process_variables, control_outputs, setpoint
        )
        
        log_line = (
            f"reset: time={time.time()} "
            f"kp={self._kp.value:.{PidProtocol.KP_DECIMAL_PLACES}f} "
            f"ki={self._ki.value:.{PidProtocol.KI_DECIMAL_PLACES}f} "
            f"kd={self._kd.value:.{PidProtocol.KD_DECIMAL_PLACES}f} "
            f"error_mean_norm={observation[0]} error_std_norm={observation[1]} "
        )
        self._env_logger.log(log_line) 

        info = {}
        return observation, info

    def close(self) -> None:
        self._backend.close()
        self._base_logger.close()

    @classmethod
    def from_config(cls, config: Config) -> "PidDeltaTuning":
        base_logger = AsyncFileLogger(log_dir=config.args.log_dir, log_file=config.args.log_file)
        backend = ExperimentalPidLoopBackend(
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
            auto_determine_setpoint=config.args.auto_determine_setpoint,
            setpoint_determination_steps=config.args.setpoint_determination_steps,
            setpoint_determination_max_value=config.args.setpoint_determination_max_value,
            setpoint_determination_factor=config.args.setpoint_determination_factor,
        )
        return cls(
            backend=backend,
            base_logger=base_logger,
            kp_min=config.args.kp_min,
            kp_max=config.args.kp_max,
            kp_start=config.args.kp_start,
            kp_delta_scale=config.args.kp_delta_scale,
            ki_min=config.args.ki_min,
            ki_max=config.args.ki_max,
            ki_start=config.args.ki_start,
            ki_delta_scale=config.args.ki_delta_scale,
            kd_min=config.args.kd_min,
            kd_max=config.args.kd_max,
            kd_start=config.args.kd_start,
            kd_delta_scale=config.args.kd_delta_scale,
            error_mean_normalization_factor=config.args.error_mean_normalization_factor,
            error_std_normalization_factor=config.args.error_std_normalization_factor,
            precision_weight=config.args.precision_weight,
            stability_weight=config.args.stability_weight,
            action_weight=config.args.action_weight,
        )