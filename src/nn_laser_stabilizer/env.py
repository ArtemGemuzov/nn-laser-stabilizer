import random
from enum import Enum
from typing import Optional, Tuple, Any, Union

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from nn_laser_stabilizer.exp_setup import ExperimentalSetupController


class Phase(Enum):
    WARMUP = "warmup"
    PRETRAIN = "pretrain"
    NORMAL = "normal"


class PidDeltaTuningEnv(gym.Env):  
    ERROR_MEAN_NORMALIZATION_FACTOR = 400
    ERROR_STD_NORMALIZATION_FACTOR = 300

    KP_MIN = 2.5
    KP_MAX = 12.5
    KP_RANGE = KP_MAX - KP_MIN
    KP_DELTA_SCALE = 0.01    
    KP_DELTA_MAX = KP_RANGE * KP_DELTA_SCALE  
    KP_START = 7.5

    KI_MIN = 0.0
    KI_MAX = 20.0
    KI_RANGE = KI_MAX - KI_MIN
    KI_DELTA_SCALE = 0.01    
    KI_DELTA_MAX = KI_RANGE * KI_DELTA_SCALE
    KI_START = 10.0
    
    PRECISION_WEIGHT = 0.4   
    STABILITY_WEIGHT = 0.4           
    ACTION_WEIGHT = 0.2              
    
    CONTROL_OUTPUT_MIN_THRESHOLD = 200.0
    CONTROL_OUTPUT_MAX_THRESHOLD = 4095.0 + 1.0

    metadata = {"render_modes": []}

    def __init__(
        self,
        setup_controller: ExperimentalSetupController,
        logger=None,
        pretrain_blocks: int = 100,
        burn_in_steps: int = 20,
    ):
        super().__init__()
        
        self.setup_controller = setup_controller
        self.logger = logger
        self._pretrain_blocks = pretrain_blocks
        self._burn_in_steps = burn_in_steps
        self._t = 0
        self._block_count = 0
        
        self.kp = self.KP_START
        self.ki = self.KI_START
        self.kd = 0.0
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def _get_phase(self) -> Phase:
        if self._block_count < self._pretrain_blocks:
            return Phase.PRETRAIN
        return Phase.NORMAL

    def _should_terminate_episode(self, control_outputs: np.ndarray) -> bool:
        mean_control_output = np.mean(control_outputs)
        
        if mean_control_output < self.CONTROL_OUTPUT_MIN_THRESHOLD:
            if self.logger is not None:
                try:
                    self.logger.log(
                        f"Episode terminated: mean_control_output={mean_control_output:.4f} "
                        f"< min_threshold={self.CONTROL_OUTPUT_MIN_THRESHOLD:.4f}"
                    )
                except Exception:
                    pass
            return True
        
        if mean_control_output > self.CONTROL_OUTPUT_MAX_THRESHOLD:
            if self.logger is not None:
                try:
                    self.logger.log(
                        f"Episode terminated: mean_control_output={mean_control_output:.4f} "
                        f"> max_threshold={self.CONTROL_OUTPUT_MAX_THRESHOLD:.4f}"
                    )
                except Exception:
                    pass
            return True
        
        return False

    def _compute_reward(self, observation: np.ndarray, action: np.ndarray) -> float:
        """
        Вычисляет награду на основе наблюдения и действия.
        
        Args:
            observation: Текущее наблюдение [error_mean_norm, error_std_norm, kp_norm, ki_norm]
            action: Выполненное действие [delta_kp_norm, delta_ki_norm]
            
        Returns:
            Значение награды
        """
        error_mean_norm, error_std_norm = observation[0], observation[1]
        
        # 1. Штраф за неточность (чем больше ошибка, тем больше штраф)
        precision_penalty = -np.abs(error_mean_norm)      # [-1, 0]
        # 2. Штраф за нестабильность (чем больше разброс, тем больше штраф)
        stability_penalty = -np.abs(error_std_norm)       # [-1, 0]
        # 3. Штраф за действие (чем больше изменение, тем больше штраф)
        action_penalty = -np.mean(np.abs(action))  # [-1, 0]
        
        total_reward = (self.PRECISION_WEIGHT * precision_penalty + 
                       self.STABILITY_WEIGHT * stability_penalty + 
                       self.ACTION_WEIGHT * action_penalty)
        
        return 2 * total_reward + 1

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        delta_kp_norm, delta_ki_norm = action[0], action[1]
        phase = self._get_phase()

        delta_kp = delta_kp_norm * self.KP_DELTA_MAX
        delta_ki = delta_ki_norm * self.KI_DELTA_MAX

        self.kp = np.clip(self.kp + delta_kp, self.KP_MIN, self.KP_MAX)
        self.ki = np.clip(self.ki + delta_ki, self.KI_MIN, self.KI_MAX)

        process_variables, control_outputs, setpoints = self.setup_controller.step(
            self.kp, self.ki, self.kd
        )
        self._t += len(process_variables)
        self._block_count += 1

        pv_window = process_variables[self._burn_in_steps:]
        sp_window = setpoints[self._burn_in_steps:]
        errors = pv_window - sp_window

        error_mean = np.mean(errors)
        error_std = np.std(errors)

        error_mean_norm = np.clip(
            error_mean / self.ERROR_MEAN_NORMALIZATION_FACTOR, -1.0, 1.0
        )
        error_std_norm = np.clip(
            error_std / self.ERROR_STD_NORMALIZATION_FACTOR, 0.0, 1.0
        )
        
        kp_norm = np.clip(
            (self.kp - self.KP_MIN) / self.KP_RANGE * 2.0 - 1.0, -1.0, 1.0
        )
        ki_norm = np.clip(
            (self.ki - self.KI_MIN) / self.KI_RANGE * 2.0 - 1.0, -1.0, 1.0
        )

        observation = np.array(
            [error_mean_norm, error_std_norm, kp_norm, ki_norm],
            dtype=np.float32
        )
        
        action_array = np.array([delta_kp_norm, delta_ki_norm], dtype=np.float32)
        reward = self._compute_reward(observation, action_array)

        if self.logger is not None:
            try: 
                log_line = (
                    f"step={self._t} phase={phase.value} block_step={self._block_count} "
                    f"kp={self.kp:.4f} ki={self.ki:.4f} kd={self.kd:.4f} "
                    f"delta_kp_norm={delta_kp_norm:.4f} delta_ki_norm={delta_ki_norm:.4f} "
                    f"error_mean={error_mean:.4f} error_std={error_std:.4f} "
                    f"error_mean_norm={error_mean_norm:.4f} error_std_norm={error_std_norm:.4f} "
                    f"reward={reward:.6f}"
                )
                self.logger.log(log_line)
            except Exception:
                pass    

        terminated = self._should_terminate_episode(control_outputs)
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.set_seed(seed)
        
        phase = Phase.WARMUP

        process_variables, control_outputs, setpoints = self.setup_controller.reset(
            kp=self.kp, ki=self.ki, kd=self.kd
        )
        self._t += len(process_variables)

        pv_window = process_variables[self._burn_in_steps:]
        sp_window = setpoints[self._burn_in_steps:]
        errors = pv_window - sp_window

        error_mean = np.mean(errors)
        error_std = np.std(errors)

        error_mean_norm = np.clip(
            error_mean / self.ERROR_MEAN_NORMALIZATION_FACTOR, -1.0, 1.0
        )
        error_std_norm = np.clip(
            error_std / self.ERROR_STD_NORMALIZATION_FACTOR, -1.0, 1.0
        )
        
        kp_norm = np.clip(
            (self.kp - self.KP_MIN) / self.KP_RANGE * 2.0 - 1.0, -1.0, 1.0
        )
        ki_norm = np.clip(
            (self.ki - self.KI_MIN) / self.KI_RANGE * 2.0 - 1.0, -1.0, 1.0
        )

        observation = np.array(
            [error_mean_norm, error_std_norm, kp_norm, ki_norm],
            dtype=np.float32
        )

        if self.logger is not None:
            try: 
                log_line = (
                    f"step={self._t} phase={phase.value} block_step={self._block_count} "
                    f"kp={self.kp:.4f} ki={self.ki:.4f} kd={self.kd:.4f} "
                    f"error_mean={error_mean:.4f} error_std={error_std:.4f} "
                    f"error_mean_norm={error_mean_norm:.4f} error_std_norm={error_std_norm:.4f} "
                )
                self.logger.log(log_line)
            except Exception:
                pass    

        info = {}
        return observation, info

    def set_seed(self, seed: Optional[int]) -> None: pass

    def close(self) -> None:
        self.setup_controller.close()


class TorchEnvWrapper(gym.Wrapper): 
    def __init__(
        self,
        env: gym.Env,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(env)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        
    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(device=self.device, dtype=self.dtype)
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.numpy()
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool, bool, dict]:
        action_np = self._to_numpy(action)
        
        observation, reward, terminated, truncated, info = self.env.step(action_np)
        
        observation_tensor = self._to_tensor(observation)
        reward_tensor = self._to_tensor(np.array(reward, dtype=np.float32))
        
        return observation_tensor, reward_tensor, terminated, truncated, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        observation, info = self.env.reset(seed=seed, options=options)
        observation_tensor = self._to_tensor(observation)
        return observation_tensor, info
    

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
