from typing import Optional, Tuple

import numpy as np
import gymnasium as gym

from nn_laser_stabilizer.plant import Plant
from nn_laser_stabilizer.logger import Logger


class PidDeltaTuningEnv(gym.Env):  
    ERROR_MEAN_NORMALIZATION_FACTOR = 400
    ERROR_STD_NORMALIZATION_FACTOR = 300

    KP_RANGE = Plant.KP_MAX - Plant.KP_MIN
    KP_DELTA_SCALE = 0.01    
    KP_DELTA_MAX = KP_RANGE * KP_DELTA_SCALE  

    KI_RANGE = Plant.KI_MAX - Plant.KI_MIN
    KI_DELTA_SCALE = 0.01    
    KI_DELTA_MAX = KI_RANGE * KI_DELTA_SCALE
    
    PRECISION_WEIGHT = 0.4   
    STABILITY_WEIGHT = 0.4           
    ACTION_WEIGHT = 0.2              

    metadata = {"render_modes": []}

    def __init__(
        self,
        plant: Plant,
        logger: Logger,
    ):
        super().__init__()
        
        self.plant: Plant = plant
        self.logger: Logger = logger
      
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
            error_mean / self.ERROR_MEAN_NORMALIZATION_FACTOR, -1.0, 1.0
        )
        error_std_norm = np.clip(
            error_std / self.ERROR_STD_NORMALIZATION_FACTOR, 0.0, 1.0
        )
        
        kp_norm = np.clip(
            (self.plant.kp - Plant.KP_MIN) / self.KP_RANGE * 2.0 - 1.0, -1.0, 1.0
        )
        ki_norm = np.clip(
            (self.plant.ki - Plant.KI_MIN) / self.KI_RANGE * 2.0 - 1.0, -1.0, 1.0
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
        
        total_reward = (self.PRECISION_WEIGHT * precision_penalty + 
                       self.STABILITY_WEIGHT * stability_penalty + 
                       self.ACTION_WEIGHT * action_penalty)
        
        return 2 * total_reward + 1

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        delta_kp_norm, delta_ki_norm = action[0], action[1]

        delta_kp = delta_kp_norm * self.KP_DELTA_MAX
        delta_ki = delta_ki_norm * self.KI_DELTA_MAX

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
        if seed is not None:
            self.set_seed(seed)

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