import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase

from nn_laser_stabilizer.envs.pid_tuning_experimental_setup import PidTuningExperimentalSetup
from nn_laser_stabilizer.envs.normalization import normalize_adc, normalize_dac
from nn_laser_stabilizer.envs.constants import DAC_MAX
from nn_laser_stabilizer.envs.control_limits import ControlLimitManager, ControlLimitConfig

class PidTuningExperimentalEnv(EnvBase):
    def __init__(self, 
                 experimental_setup : PidTuningExperimentalSetup, 
                 action_spec,
                 observation_spec,
                 reward_spec,
                 reward_func,
                 fixed_kp: float | None = None,
                 fixed_ki: float | None = None,
                 fixed_kd: float | None = None,
                 default_min: float | None = None,
                 default_max: float | None = None,
                 force_min_value: float | None = None,
                 force_condition_threshold: float | None = None,
                 enforcement_steps: int | None = None
    ):
        super().__init__()

        self.experimental_setup = experimental_setup

        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec

        self.reward_func = reward_func

        config = ControlLimitConfig(
            default_min=0.0 if default_min is None else float(default_min),
            default_max=DAC_MAX if default_max is None else float(default_max),
            force_min_value=float(force_min_value),
            force_condition_threshold=float(force_condition_threshold),
            enforcement_steps=int(enforcement_steps),
        )
        self._control_limits = ControlLimitManager(config)

        if fixed_kp is not None and fixed_ki is not None and fixed_kd is not None:
            self._fixed_action = (float(fixed_kp), float(fixed_ki), float(fixed_kd))
        else:
            self._fixed_action = None

    def _step(self, tensordict: TensorDict) -> TensorDict:
        kp, ki, kd = tensordict["action"].tolist()
        if self._fixed_action is not None:
            kp, ki, kd = self._fixed_action
        applied_min, applied_max = self._control_limits.get_limits_for_step()

        process_variable, control_output, setpoint = self.experimental_setup.step(kp, ki, kd, applied_min, applied_max)

        # TODO: возможно, тут стоит отправлять сигналы в цикле, избегая вывод информации наружу
        self._control_limits.apply_rule(control_output)

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
    