import time
import random
from collections import deque
from enum import Enum

import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase

from nn_laser_stabilizer.envs.pid_tuning_experimental_setup import PidTuningExperimentalSetup
from nn_laser_stabilizer.envs.constants import DEFAULT_KP, DEFAULT_KI, DEFAULT_KD, KP_MIN, KP_MAX, KI_MIN, KI_MAX, KD_MIN, KD_MAX
from nn_laser_stabilizer.envs.normalization import denormalize_kp, denormalize_ki, denormalize_kd, normalize_kp, normalize_ki, normalize_kd


# TODO: занести в класс
ERROR_MEAN_NORMALIZATION_FACTOR = 2.0
ERROR_STD_NORMALIZATION_FACTOR = 20.0


class Phase(Enum):
    WARMUP = "warmup"
    PRETRAIN = "pretrain"
    NORMAL = "normal"
    

class PidTuningExperimentalEnv(EnvBase):
    def __init__(self, 
                 experimental_setup : PidTuningExperimentalSetup, 
                 action_spec,
                 observation_spec,
                 reward_spec,
                 reward_func,
                 logger=None,
                 warmup_steps: int = 1000,
                 pretrain_blocks: int = 100,
                 block_size: int = 100,
                 burn_in_steps: int = 20,
                 force_min_value: float = 2000.0,
                 force_max_value: float = 4095.0,
                 default_min: float = 0.0,
                 default_max: float = 4095.0,
    ):
        super().__init__()

        self.experimental_setup = experimental_setup

        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec

        self.reward_func = reward_func
        self._force_min_value = force_min_value
        self._force_max_value = force_max_value
        self._default_min = default_min
        self._default_max = default_max

        self.logger = logger
        self._t = 0
      
        self._warmup_steps = int(warmup_steps)
        self._pretrain_blocks = int(pretrain_blocks)
        self._block_size = int(block_size)
        self._burn_in_steps = int(burn_in_steps)
        
        if self._block_size < 3:
            raise ValueError(
                f"block_size must be at least 3 (got block_size={self._block_size})"
            )
        
        if self._burn_in_steps >= self._block_size:
            raise ValueError(
                f"burn_in_steps must be less than block_size (got burn_in_steps={self._burn_in_steps}, block_size={self._block_size})"
            )
    
        self.window_size = self._block_size - self._burn_in_steps
        self.errors = deque(maxlen=self.window_size)  
        self.rewards = deque(maxlen=self.window_size)  

    def _get_phase(self) -> Phase:
        if self._t < self._warmup_steps:
            return Phase.WARMUP
        
        steps_after_warmup = self._t - self._warmup_steps
        blocks_after_warmup = steps_after_warmup // self._block_size
        
        if blocks_after_warmup < self._pretrain_blocks:
            return Phase.PRETRAIN
        return Phase.NORMAL

    def _log_step(
        self,
        *,
        kp: float, ki: float, kd: float,
        process_variable: float, control_output,
        u_min: float, u_max: float,
        phase: str,
        block_step: int,
    ) -> None:
        if self.logger is None:
            return
        try:
            now = time.time()
            log_line = (
                f"step={self._t} time={now:.6f} phase={phase} "
                f"block_step={block_step} "
                f"kp={kp:.4f} ki={ki:.4f} kd={kd:.4f} "
                f"process_variable={process_variable:.1f} control_output={control_output:.1f} "
                f"u_min={u_min:.1f} u_max={u_max:.1f}"
            )
            self.logger.log(log_line)
        except Exception:
            pass

    def _step(self, tensordict: TensorDict) -> TensorDict:
        agent_kp_norm, agent_ki_norm, agent_kd_norm = tensordict["action"].tolist()

        phase = self._get_phase()

        match phase:
            case Phase.WARMUP:
                kp, ki, kd = float(DEFAULT_KP), float(DEFAULT_KI), float(DEFAULT_KD)
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

        for block_iteration in range(self._block_size):
            if phase == Phase.WARMUP:
                applied_min = self._force_min_value
                applied_max = self._force_max_value
            else:
                applied_min = self._default_min
                applied_max = self._default_max

            process_variable, control_output, setpoint = self.experimental_setup.step(
                kp, ki, kd, applied_min, applied_max
            )

            self._log_step(
                kp=kp, ki=ki, kd=kd,
                process_variable=process_variable, control_output=control_output,
                u_min=applied_min, u_max=applied_max,
                phase=phase.value,
                block_step=block_iteration,
            )
            
            if block_iteration >= self._burn_in_steps:
                error = process_variable - setpoint
                step_reward = self.reward_func(process_variable, setpoint)
                
                self.errors.append(error)
                self.rewards.append(step_reward)

            self._t += 1

        error_mean = sum(self.errors) / self.window_size
        error_variance = sum((error - error_mean) ** 2 for error in self.errors) / self.window_size
        error_std = error_variance ** 0.5

        error_mean_norm = error_mean / ERROR_MEAN_NORMALIZATION_FACTOR
        error_std_norm = error_std / ERROR_STD_NORMALIZATION_FACTOR

        observation = torch.tensor(
            [error_mean_norm, error_std_norm],
            dtype=torch.float32,
            device=self.device
        )
        
        reward = sum(self.rewards) / len(self.rewards)
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

        process_variable, control_output, setpoint = self.experimental_setup.reset()

        if self.logger is not None:
            try:
                now = time.time()
                log_line = f"reset time={now:.6f} process_variable={process_variable:.8f} control_output={control_output:.8f} setpoint={setpoint:.8f}"
                self.logger.log(log_line)
            except Exception:
                pass

        error = process_variable - setpoint
        error_mean = error  
        error_std = 0.0     
        
        observation = torch.tensor(
            [error_mean, error_std],
            dtype=torch.float32,
            device=self.device
        )

        return TensorDict({"observation": observation}, batch_size=[])

    def _set_seed(self, seed: int):
        self.experimental_setup.set_seed(seed)

    def set_state(self, state):
        pass

    def forward(self, tensordict):
        if "observation" not in tensordict:
            tensordict = self.reset()
        return self.step(tensordict)
    