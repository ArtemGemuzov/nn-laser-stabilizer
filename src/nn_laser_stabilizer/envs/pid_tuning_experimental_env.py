import torch
import time
from tensordict import TensorDict
from torchrl.envs import EnvBase

from nn_laser_stabilizer.envs.pid_tuning_experimental_setup import PidTuningExperimentalSetup
from nn_laser_stabilizer.envs.normalization import normalize_adc, normalize_dac
from nn_laser_stabilizer.envs.constants import DAC_MAX
from nn_laser_stabilizer.envs.control_limit_manager import ControlLimitManager
from nn_laser_stabilizer.envs.fixed_pid_manager import FixedPidManager
from nn_laser_stabilizer.logging.async_file_logger import AsyncFileLogger

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

    def _log_step(
        self,
        *,
        kp: float, ki: float, kd: float,
        process_variable: float, control_output: float, setpoint: float,
        u_min: float, u_max: float,
    ) -> None:
        if self.logger is None:
            return
        try:
            now = time.time()
            log_line = (
                f"step={self._t} time={now:.6f} "
                f"kp={kp:.8f} ki={ki:.8f} kd={kd:.8f} "
                f"process_variable={process_variable:.8f} control_output={control_output:.8f} setpoint={setpoint:.8f} "
                f"u_min={u_min:.8f} u_max={u_max:.8f}"
            )
            self.logger.log(log_line)
        except Exception:
            pass

    def _step(self, tensordict: TensorDict) -> TensorDict:
        agent_kp, agent_ki, agent_kd = tensordict["action"].tolist()
        kp, ki, kd = self._fixed_pid_manager.get_coefficients(agent_kp, agent_ki, agent_kd)

        applied_min, applied_max = self._control_limits.get_limits_for_step()

        process_variable, control_output, setpoint = self.experimental_setup.step(kp, ki, kd, applied_min, applied_max)

        self._log_step(
            kp=kp,
            ki=ki,
            kd=kd,
            process_variable=process_variable,
            control_output=control_output,
            setpoint=setpoint,
            u_min=applied_min,
            u_max=applied_max,
        )

        # TODO: возможно, тут стоит отправлять сигналы в цикле, избегая вывод информации наружу
        self._control_limits.apply_rule(control_output)
        self._t += 1

        # лежит в [-1; 1]
        process_variable_norm = normalize_adc(process_variable)
        setpoint_norm = normalize_adc(setpoint)
        control_output_norm = normalize_dac(control_output)

        observation = torch.tensor(
            [process_variable_norm, control_output_norm, setpoint_norm],
            dtype=torch.float32,
            device=self.device
        )

        reward = self.reward_func(process_variable, setpoint)

        done = False  # TODO задать условие завершения

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

        process_variable, control_output, setpoint = self.experimental_setup.reset()

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
    