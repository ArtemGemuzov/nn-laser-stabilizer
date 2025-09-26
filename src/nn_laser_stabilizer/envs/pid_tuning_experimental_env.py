import time
import random
from collections import deque
from enum import Enum

import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase

from nn_laser_stabilizer.envs.pid_tuning_experimental_setup import PidTuningExperimentalSetup
from nn_laser_stabilizer.envs.normalization import normalize_adc, normalize_dac
from nn_laser_stabilizer.envs.constants import DEFAULT_KP, DEFAULT_KI, DEFAULT_KD, KP_MIN, KP_MAX, KI_MIN, KI_MAX, KD_MIN, KD_MAX
from nn_laser_stabilizer.envs.control_limit_manager import ControlLimitManager
from nn_laser_stabilizer.envs.fixed_pid_manager import FixedPidManager


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
                 control_limits: ControlLimitManager,
                 fixed_pid: FixedPidManager,
                 logger=None,
                 warmup_steps: int = 1000,
                 pretrain_steps: int = 10_000,
                 block_size: int = 100,
                 burn_in_steps: int = 20,
    ):
        super().__init__()

        self.experimental_setup = experimental_setup

        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec

        self.reward_func = reward_func
        self._control_limits = control_limits
        self._fixed_pid_manager = fixed_pid

        self.logger = logger
        self._t = 0
      
        self._warmup_steps = int(warmup_steps)
        self._pretrain_steps = int(pretrain_steps)
        self._block_size = int(block_size)
        self._burn_in_steps = int(burn_in_steps)
        if self._burn_in_steps >= self._block_size:
            raise ValueError(
                f"burn_in_steps must be less than block_size (got burn_in_steps={self._burn_in_steps}, block_size={self._block_size})"
            )
    
        self.window_size = self._block_size - self._burn_in_steps
        self.process_variables = deque(maxlen=self.window_size)
        self.control_outputs = deque(maxlen=self.window_size)
        self.setpoints = deque(maxlen=self.window_size)

    def _get_phase(self) -> Phase:
        if self._t < self._warmup_steps:
            return Phase.WARMUP
        if self._t < self._warmup_steps + self._pretrain_steps:
            return Phase.PRETRAIN
        return Phase.NORMAL

    def _log_step(
        self,
        *,
        kp: float, ki: float, kd: float,
        process_variable: float, control_output: float, setpoint: float,
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
                f"process_variable={process_variable:.1f} control_output={control_output:.1f} setpoint={setpoint:.1f} "
                f"u_min={u_min:.1f} u_max={u_max:.1f}"
            )
            self.logger.log(log_line)
        except Exception:
            pass

    def _step(self, tensordict: TensorDict) -> TensorDict:
        agent_kp, agent_ki, agent_kd = tensordict["action"].tolist()

        phase = self._get_phase()

        match phase:
            case Phase.WARMUP:
                kp, ki, kd = float(DEFAULT_KP), float(DEFAULT_KI), float(DEFAULT_KD)
            case Phase.PRETRAIN:
                kp = random.uniform(KP_MIN, KP_MAX)
                ki = random.uniform(KI_MIN, KI_MAX)
                kd = random.uniform(KD_MIN, KD_MAX)
            case Phase.NORMAL:
                kp, ki, kd = self._fixed_pid_manager.get_coefficients(agent_kp, agent_ki, agent_kd)
            case _:
                raise ValueError(f"Unknown phase: {phase}")

        for block_iteration in range(self._block_size):
            applied_min, applied_max = self._control_limits.get_limits_for_step()
            if phase == Phase.WARMUP:
                applied_min = self._control_limits.config.force_min_value
                applied_max = self._control_limits.config.default_max

            process_variable, control_output, setpoint = self.experimental_setup.step(
                kp, ki, kd, applied_min, applied_max
            )

            self._log_step(
                kp=kp, ki=ki, kd=kd,
                process_variable=process_variable, control_output=control_output, setpoint=setpoint,
                u_min=applied_min, u_max=applied_max,
                phase=phase.value,
                block_step=block_iteration,
            )

            self._control_limits.apply_rule(control_output)

            if block_iteration >= self._burn_in_steps:
                self.process_variables.append(process_variable)
                self.control_outputs.append(control_output)
                self.setpoints.append(setpoint)

            self._t += 1

        average_process_variable = sum(self.process_variables) / len(self.process_variables)
        average_control_output = sum(self.control_outputs) / len(self.control_outputs)
        average_setpoint = sum(self.setpoints) / len(self.setpoints)

        process_variable_norm = normalize_adc(average_process_variable)
        setpoint_norm = normalize_adc(average_setpoint)
        control_output_norm = normalize_dac(average_control_output)

        observation = torch.tensor(
            [process_variable_norm, control_output_norm, setpoint_norm],
            dtype=torch.float32,
            device=self.device
        )
        reward = self.reward_func(average_process_variable, average_setpoint)
        done = False  # TODO задать условие завершения

        if self.logger is not None:
            try:
                now = time.time()
                log_line = (
                    f"step={self._t} time={now:.6f} phase={phase.value} "
                    f"block_step=final "
                    f"kp={kp:.4f} ki={ki:.4f} kd={kd:.4f} "
                    f"process_variable_norm={process_variable_norm:.4f} control_output_norm={control_output_norm:.4f} setpoint_norm={setpoint_norm:.4f} "
                    f"reward={reward:.6f}"
                )
                self.logger.log(log_line)
            except Exception:
                pass

        tensordict.set("action", torch.tensor([kp, ki, kd], dtype=torch.float32, device=self.device))

        return TensorDict(
            {
                "observation": observation,
                "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
            },
            batch_size=[]
        )

    def _reset(self, unused: TensorDict | None = None) -> TensorDict:
        self._control_limits.reset()
        self._t = 0

        process_variable, control_output, setpoint = self.experimental_setup.reset()

        if self.logger is not None:
            try:
                now = time.time()
                log_line = f"reset time={now:.6f} process_variable={process_variable:.8f} control_output={control_output:.8f} setpoint={setpoint:.8f}"
                self.logger.log(log_line)
            except Exception:
                pass

        process_variable_norm = normalize_adc(process_variable)
        setpoint_norm = normalize_adc(setpoint)
        control_output_norm = normalize_dac(control_output)

        observation = torch.tensor(
            [process_variable_norm, control_output_norm, setpoint_norm],
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
    