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

    ERROR_MEAN_NORMALIZATION_FACTOR = 1.0
    ERROR_STD_NORMALIZATION_FACTOR = 1.0

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
        agent_kp_norm, agent_ki_norm, agent_kd_norm = tensordict["action"].tolist()

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
        
        rewards = self.reward_func(pv_window, sp_window)
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
    ERROR_MEAN_NORMALIZATION_FACTOR = 1.0
    ERROR_STD_NORMALIZATION_FACTOR = 1.0
    DELTA_PENALTY = 0.01  

    KP_MIN = 0.0
    KP_MAX = 10.0

    KI = 11.0
    KD = 0.002
    DEFAULT_KP = 3.5


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
        self.kp = self.DEFAULT_KP

    def _get_phase(self) -> Phase:
        if self._block_count < self._pretrain_blocks:
            return Phase.PRETRAIN
        return Phase.NORMAL

    def _compute_reward(self, errors, delta_kp):
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        penalty = self.DELTA_PENALTY * (delta_kp ** 2)
        reward = -(abs(error_mean) + error_std + penalty)
        return reward, error_mean, error_std

    def _step(self, tensordict: TensorDict) -> TensorDict:
        agent_delta_norm = tensordict["action"].item() 
        phase = self._get_phase()

        delta_kp = agent_delta_norm * 0.01 * (self.KP_MAX - self.KP_MIN)  
        if phase == Phase.PRETRAIN:
            delta_kp = random.uniform(-0.01 * (self.KP_MAX - self.KP_MIN),
                                      0.01 * (self.KP_MAX - self.KP_MIN))

        self.kp = np.clip(self.kp + delta_kp, self.KP_MIN, self.KP_MAX)

        process_variables, control_outputs, setpoints = self.setup_controller.step(
            self.kp, self.KI, self.KD
        )
        self._t += len(process_variables)
        self._block_count += 1

        pv_window = process_variables[self._burn_in_steps:]
        sp_window = setpoints[self._burn_in_steps:]
        errors = pv_window - sp_window

        error_mean = np.mean(errors)
        error_std = np.std(errors)

        error_mean_norm = error_mean / self.ERROR_MEAN_NORMALIZATION_FACTOR
        error_std_norm = error_std / self.ERROR_STD_NORMALIZATION_FACTOR

        reward, error_mean, error_std = self._compute_reward(errors, delta_kp)

        observation = torch.tensor(
            [error_mean_norm,
             error_std_norm,
             self.kp],
            dtype=torch.float32,
            device=self.device
        )

        done = False

        if self.logger is not None:
            try: 
                log_line = (
                    f"step={self._t} phase={phase.value} "
                    f"block_step=final "
                    f"kp={self.kp} ki={self.KI} kd={self.KD} "
                    f"error_mean={error_mean:.4f} error_std={error_std:.4f} "
                    f"error_mean_norm={error_mean_norm:.4f} error_std_norm={error_std_norm:.4f} "
                )
                self.logger.log(log_line)
            except Exception:
                pass    

        tensordict.set("action", torch.tensor([agent_delta_norm], dtype=torch.float32, device=self.device))

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
        self.kp = self.DEFAULT_KP
        phase = Phase.WARMUP

        process_variables, control_outputs, setpoints = self.setup_controller.reset(
            kp=self.kp, ki=self.KI, kd=self.KD
        )
        self._t += len(process_variables)

        pv_window = process_variables[self._burn_in_steps:]
        sp_window = setpoints[self._burn_in_steps:]
        errors = pv_window - sp_window

        error_mean = np.mean(errors)
        error_std = np.std(errors)

        error_mean_norm = error_mean / self.ERROR_MEAN_NORMALIZATION_FACTOR
        error_std_norm = error_std / self.ERROR_STD_NORMALIZATION_FACTOR

        observation = torch.tensor(
            [error_mean_norm,
             error_std_norm,
             self.kp],
            dtype=torch.float32,
            device=self.device
        )

        if self.logger is not None:
            try: 
                log_line = (
                    f"step={self._t} phase={phase.value} "
                    f"block_step=final "
                    f"kp={self.kp} ki={self.KI} kd={self.KD} "
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

