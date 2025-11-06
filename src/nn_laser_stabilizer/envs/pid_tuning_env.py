import random
from enum import Enum

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase

from nn_laser_stabilizer.envs.experimental_setup_protocol import ExperimentalSetupProtocol
from nn_laser_stabilizer.envs.normalizer import Normalizer


class Phase(Enum):
    WARMUP = "warmup"
    PRETRAIN = "pretrain"
    NORMAL = "normal"
    

class PidTuningEnv(EnvBase):

    ERROR_MEAN_NORMALIZATION_FACTOR = 250.0
    ERROR_STD_NORMALIZATION_FACTOR = 200.0

    def __init__(self, 
                 setup_controller: ExperimentalSetupProtocol, 
                 action_spec,
                 observation_spec,
                 reward_spec,
                 reward_func,
                 normalizer: Normalizer,
                 logger=None,
                 pretrain_blocks: int = 100,
                 burn_in_steps: int = 20,
    ):
        super().__init__()

        self.setup_controller = setup_controller

        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec

        self.reward_func = reward_func
        self.normalizer = normalizer

        self.logger = logger
        self._t = 0
      
        self._pretrain_blocks = int(pretrain_blocks)
        self._burn_in_steps = int(burn_in_steps)
        self._block_count = 0  

    def _get_phase(self) -> Phase:
        if self._block_count < self._pretrain_blocks:
            return Phase.PRETRAIN
        return Phase.NORMAL

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"].tolist()
        agent_kp_norm, agent_ki_norm, agent_kd_norm = action

        phase = self._get_phase()

        match phase:
            case Phase.PRETRAIN:
                kp_range = self.normalizer.kp_max - self.normalizer.kp_min
                ki_range = self.normalizer.ki_max - self.normalizer.ki_min
                kd_range = self.normalizer.kd_max - self.normalizer.kd_min
                
                kp = self.normalizer.kp_min + random.uniform(0.25 * kp_range, 0.75 * kp_range)
                ki = self.normalizer.ki_min + random.uniform(0.25 * ki_range, 0.75 * ki_range)
                kd = self.normalizer.kd_min + random.uniform(0.25 * kd_range, 0.75 * kd_range)
            case Phase.NORMAL:
                kp = self.normalizer.denormalize_kp(agent_kp_norm)
                ki = self.normalizer.denormalize_ki(agent_ki_norm)
                kd = self.normalizer.denormalize_kd(agent_kd_norm)
            case _:
                raise ValueError(f"Unknown phase: {phase}")

        process_variables, control_outputs, setpoints = self.setup_controller.step(kp, ki, kd)
        self._t += len(process_variables)
        self._block_count += 1
        
        pv_window = process_variables[self._burn_in_steps:]
        sp_window = setpoints[self._burn_in_steps:]
        
        errors = pv_window - sp_window
         
        error_mean = np.mean(errors)
        error_std = np.std(errors)

        error_mean_norm = error_mean / self.ERROR_MEAN_NORMALIZATION_FACTOR
        error_std_norm = error_std / self.ERROR_STD_NORMALIZATION_FACTOR

        kp_norm = self.normalizer.normalize_kp(kp)
        ki_norm = self.normalizer.normalize_ki(ki)
        kd_norm = self.normalizer.normalize_kd(kd)
        
        observation = torch.tensor(
            [error_mean_norm, error_std_norm, kp_norm, ki_norm, kd_norm],
            dtype=torch.float32,
            device=self.device
        )
        
        rewards = self.reward_func(pv_window, sp_window, action)
        reward = np.mean(rewards)
        
        done = False # False, потому что при True насильно вызывается reset

        if self.logger is not None:
            try: 
                log_line = (
                    f"step={self._t} phase={phase.value} "
                    f"block_step=final "
                    f"kp={kp:.4f} ki={ki:.4f} kd={kd:.4f} "
                    f"kp_norm={kp_norm:.4f} ki_norm={ki_norm:.4f} kd_norm={kd_norm:.4f} "
                    f"error_mean={error_mean:.4f} error_std={error_std:.4f} "
                    f"error_mean_norm={error_mean_norm:.4f} error_std_norm={error_std_norm:.4f} "
                    f"reward={reward:.6f}"
                )
                self.logger.log(log_line)
            except Exception:
                pass

        tensordict.set("action", torch.tensor([kp_norm, ki_norm, kd_norm], dtype=torch.float32, device=self.device))

        return TensorDict(
            {
                "observation": observation,
                "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
            },
            batch_size=[]
        )

    def _reset(self, unused: TensorDict | None = None) -> TensorDict:
        self._t = 0
        self._block_count = 0

        phase = Phase.WARMUP

        process_variables, control_outputs, setpoints = self.setup_controller.reset(
            kp=self.normalizer.default_kp,
            ki=self.normalizer.default_ki,
            kd=self.normalizer.default_kd
        )
        warmup_steps = len(process_variables)
        self._t += warmup_steps

        pv_window = process_variables[self._burn_in_steps:]
        sp_window = setpoints[self._burn_in_steps:]
        
        errors = pv_window - sp_window

        error_mean = np.mean(errors)
        error_std = np.std(errors)
        
        error_mean_norm = error_mean / self.ERROR_MEAN_NORMALIZATION_FACTOR
        error_std_norm = error_std / self.ERROR_STD_NORMALIZATION_FACTOR

        kp_norm = self.normalizer.normalize_kp(self.normalizer.default_kp)
        ki_norm = self.normalizer.normalize_ki(self.normalizer.default_ki)
        kd_norm = self.normalizer.normalize_kd(self.normalizer.default_kd)

        if self.logger is not None:
            try: 
                log_line = (
                    f"step={self._t} phase={phase.value} "
                    f"block_step=final "
                    f"kp={self.normalizer.default_kp:.4f} ki={self.normalizer.default_ki:.4f} kd={self.normalizer.default_kd:.4f} "
                    f"error_mean={error_mean:.4f} error_std={error_std:.4f} "
                    f"error_mean_norm={error_mean_norm:.4f} error_std_norm={error_std_norm:.4f} "
                )
                self.logger.log(log_line)
            except Exception:
                pass    
        
        observation = torch.tensor(
            [error_mean_norm, error_std_norm, kp_norm, ki_norm, kd_norm],
            dtype=torch.float32,
            device=self.device
        )

        return TensorDict({"observation": observation}, batch_size=[])

    def _set_seed(self, seed: int):
        self.setup_controller.set_seed(seed)

    def set_state(self, state):
        pass

    def forward(self, tensordict):
        if "observation" not in tensordict:
            tensordict = self.reset()
        return self.step(tensordict)


class PidDeltaTuningEnv(EnvBase):
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

    KD_MIN = 0.0
    KD_MAX = 0.01
    KD_RANGE = KD_MAX - KD_MIN
    KD_DELTA_SCALE = 0.01
    KD_DELTA_MAX = KD_RANGE * KD_DELTA_SCALE
    KD_START = 0.002
    
    PRECISION_WEIGHT = 0.4   
    STABILITY_WEIGHT = 0.4           
    ACTION_WEIGHT = 0.2              
    
    CONTROL_OUTPUT_MIN_THRESHOLD = 500.0
    CONTROL_OUTPUT_MAX_THRESHOLD = 3500.0

    def __init__(self,
                 setup_controller,
                 action_spec,
                 observation_spec,
                 reward_spec,
                 reward_func=None,
                 normalizer=None,
                 logger=None,
                 pretrain_blocks: int = 100,
                 burn_in_steps: int = 20,
                 **kwargs):
        super().__init__()

        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec

        self.setup_controller = setup_controller
        self.logger = logger
        self._pretrain_blocks = pretrain_blocks
        self._burn_in_steps = burn_in_steps
        self._t = 0
        self._block_count = 0
        
        self.kp = self.KP_START
        self.ki = self.KI_START
        self.kd = self.KD_START

        self._has_been_called_once = False

    def _get_phase(self) -> Phase:
        if self._block_count < self._pretrain_blocks:
            return Phase.PRETRAIN
        return Phase.NORMAL

    def _should_terminate_episode(self, control_outputs) -> bool:
        if len(control_outputs) == 0:
            return False
        
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

    def _compute_reward(self, observation, action):
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

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"].tolist()
        delta_kp_norm, delta_ki_norm, delta_kd_norm = action
        phase = self._get_phase()

        if phase == Phase.PRETRAIN:
            delta_kp_norm = np.clip(np.random.normal(0, 1), -1, 1)
            delta_ki_norm = np.clip(np.random.normal(0, 1), -1, 1)
            delta_kd_norm = np.clip(np.random.normal(0, 1), -1, 1)
        
        delta_kp = delta_kp_norm * self.KP_DELTA_MAX
        delta_ki = delta_ki_norm * self.KI_DELTA_MAX
        delta_kd = delta_kd_norm * self.KD_DELTA_MAX

        self.kp = np.clip(self.kp + delta_kp, self.KP_MIN, self.KP_MAX)
        self.ki = np.clip(self.ki + delta_ki, self.KI_MIN, self.KI_MAX)
        self.kd = np.clip(self.kd + delta_kd, self.KD_MIN, self.KD_MAX)

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

        error_mean_norm = np.clip(error_mean / self.ERROR_MEAN_NORMALIZATION_FACTOR, -1.0, 1.0)
        error_std_norm = np.clip(error_std / self.ERROR_STD_NORMALIZATION_FACTOR, 0.0, 1.0)
        kp_norm =  np.clip((self.kp - self.KP_MIN) / self.KP_RANGE * 2.0 - 1.0, -1.0, 1.0)
        ki_norm =  np.clip((self.ki - self.KI_MIN) / self.KI_RANGE * 2.0 - 1.0, -1.0, 1.0)
        kd_norm =  np.clip((self.kd - self.KD_MIN) / self.KD_RANGE * 2.0 - 1.0, -1.0, 1.0)

        observation = np.array([error_mean_norm, error_std_norm, kp_norm, ki_norm, kd_norm], dtype=np.float32)
        action = np.array([delta_kp_norm, delta_ki_norm, delta_kd_norm], dtype=np.float32)
        reward = self._compute_reward(observation, action)

        if self.logger is not None:
            try: 
                log_line = (
                    f"step={self._t} phase={phase.value} block_step={self._block_count} "
                    f"kp={self.kp:.4f} ki={self.ki:.4f} kd={self.kd:.6f} "
                    f"delta_kp_norm={delta_kp_norm:.4f} delta_ki_norm={delta_ki_norm:.4f} delta_kd_norm={delta_kd_norm:.4f} "
                    f"error_mean={error_mean:.4f} error_std={error_std:.4f} "
                    f"error_mean_norm={error_mean_norm:.4f} error_std_norm={error_std_norm:.4f} "
                    f"reward={reward:.6f}"
                )
                self.logger.log(log_line)
            except Exception:
                pass    

        done = self._should_terminate_episode(control_outputs)

        tensordict.set("action", torch.tensor([delta_kp_norm, delta_ki_norm, delta_kd_norm], dtype=torch.float32, device=self.device))

        return TensorDict(
            {
                "observation": torch.tensor(observation, dtype=torch.float32, device=self.device),
                "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
            },
            batch_size=[]
        )

    def _reset(self, unused: TensorDict | None = None) -> TensorDict:
        if not self._has_been_called_once:
            self._has_been_called_once = True
            observation = torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=torch.float32,
                device=self.device
            )
            return TensorDict({"observation": observation}, batch_size=[])  
        
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

        error_mean_norm = np.clip(error_mean / self.ERROR_MEAN_NORMALIZATION_FACTOR, -1.0, 1.0)
        error_std_norm = np.clip(error_std / self.ERROR_STD_NORMALIZATION_FACTOR, -1.0, 1.0)
        kp_norm =  np.clip((self.kp - self.KP_MIN) / self.KP_RANGE * 2.0 - 1.0, -1.0, 1.0)
        ki_norm =  np.clip((self.ki - self.KI_MIN) / self.KI_RANGE * 2.0 - 1.0, -1.0, 1.0)
        kd_norm =  np.clip((self.kd - self.KD_MIN) / self.KD_RANGE * 2.0 - 1.0, -1.0, 1.0)

        observation = torch.tensor(
            [error_mean_norm,
             error_std_norm,
             kp_norm,
             ki_norm,
             kd_norm],
            dtype=torch.float32,
            device=self.device
        )

        if self.logger is not None:
            try: 
                log_line = (
                    f"step={self._t} phase={phase.value} block_step={self._block_count} "
                    f"kp={self.kp:.4f} ki={self.ki:.4f} kd={self.kd:.6f} "
                    f"error_mean={error_mean:.4f} error_std={error_std:.4f} "
                    f"error_mean_norm={error_mean_norm:.4f} error_std_norm={error_std_norm:.4f} "
                )
                self.logger.log(log_line)
            except Exception:
                pass    

        return TensorDict({"observation": observation}, batch_size=[])

    def _set_seed(self, seed: int):
        self.setup_controller.set_seed(seed)

    def set_state(self, state):
        pass

    def forward(self, tensordict):
        if "observation" not in tensordict:
            tensordict = self.reset()
        return self.step(tensordict)

