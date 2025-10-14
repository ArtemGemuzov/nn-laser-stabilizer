import random
from enum import Enum

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase

from nn_laser_stabilizer.envs.experimental_setup_controller import ExperimentalSetupController
from nn_laser_stabilizer.envs.constants import DEFAULT_KP, DEFAULT_KI, DEFAULT_KD, KP_MIN, KP_MAX, KI_MIN, KI_MAX, KD_MIN, KD_MAX
from nn_laser_stabilizer.envs.normalization import denormalize_kp, denormalize_ki, denormalize_kd, normalize_kp, normalize_ki, normalize_kd


# TODO: занести в класс
ERROR_MEAN_NORMALIZATION_FACTOR = 2.0
ERROR_STD_NORMALIZATION_FACTOR = 20.0


class Phase(Enum):
    WARMUP = "warmup"
    PRETRAIN = "pretrain"
    NORMAL = "normal"
    

class PidTuningEnv(EnvBase):
    def __init__(self, 
                 setup_controller: ExperimentalSetupController, 
                 action_spec,
                 observation_spec,
                 reward_spec,
                 reward_func,
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
                kp_range = KP_MAX - KP_MIN
                ki_range = KI_MAX - KI_MIN
                kd_range = KD_MAX - KD_MIN
                
                kp = KP_MIN + random.uniform(0.25 * kp_range, 0.75 * kp_range)
                ki = KI_MIN + random.uniform(0.25 * ki_range, 0.75 * ki_range)
                kd = KD_MIN + random.uniform(0.25 * kd_range, 0.75 * kd_range)
            case Phase.NORMAL:
                kp = denormalize_kp(agent_kp_norm)
                ki = denormalize_ki(agent_ki_norm)
                kd = denormalize_kd(agent_kd_norm)
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

        error_mean_norm = error_mean / ERROR_MEAN_NORMALIZATION_FACTOR
        error_std_norm = error_std / ERROR_STD_NORMALIZATION_FACTOR

        observation = torch.tensor(
            [error_mean_norm, error_std_norm],
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
                    f"error_mean={error_mean:.4f} error_std={error_std:.4f} "
                    f"error_mean_norm={error_mean_norm:.4f} error_std_norm={error_std_norm:.4f} "
                    f"reward={reward:.6f}"
                )
                self.logger.log(log_line)
            except Exception:
                pass

        kp_norm = normalize_kp(kp)
        ki_norm = normalize_ki(ki) 
        kd_norm = normalize_kd(kd)
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

        process_variables, control_outputs, setpoints = self.setup_controller.reset()
        warmup_steps = len(process_variables)
        self._t += warmup_steps

        pv_window = process_variables[self._burn_in_steps:]
        sp_window = setpoints[self._burn_in_steps:]
        
        errors = pv_window - sp_window

        error_mean = np.mean(errors)
        error_std = np.std(errors)
        
        error_mean_norm = error_mean / ERROR_MEAN_NORMALIZATION_FACTOR
        error_std_norm = error_std / ERROR_STD_NORMALIZATION_FACTOR

        if self.logger is not None:
            try: 
                log_line = (
                    f"step={self._t} phase={phase.value} "
                    f"block_step=final "
                    f"kp={DEFAULT_KP:.4f} ki={DEFAULT_KI:.4f} kd={DEFAULT_KD:.4f} "
                    f"error_mean={error_mean:.4f} error_std={error_std:.4f} "
                    f"error_mean_norm={error_mean_norm:.4f} error_std_norm={error_std_norm:.4f} "
                )
                self.logger.log(log_line)
            except Exception:
                pass    
        
        observation = torch.tensor(
            [error_mean_norm, error_std_norm],
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
