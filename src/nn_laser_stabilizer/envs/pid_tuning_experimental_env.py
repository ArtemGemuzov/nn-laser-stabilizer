import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase

from nn_laser_stabilizer.envs.pid_tuning_experimental_setup import PidTuningExperimentalSetup
from nn_laser_stabilizer.envs.normalization import normalize_adc, normalize_dac
from nn_laser_stabilizer.envs.constants import DAC_MAX

class PidTuningExperimentalEnv(EnvBase):
    def __init__(self, 
                 experimental_setup : PidTuningExperimentalSetup, 
                 action_spec,
                 observation_spec,
                 reward_spec,
                 reward_func,
                 fixed_kp: float | None = None,
                 fixed_ki: float | None = None,
                 fixed_kd: float | None = None
    ):
        super().__init__()

        self.experimental_setup = experimental_setup

        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec

        self.reward_func = reward_func

        self._default_min = 0.0
        self._default_max = DAC_MAX

        self._current_min = self._default_min
        self._current_max = self._default_max

        # TODO: вынести в константы или конфиг
        self._force_min_value = 2000.0
        self._force_condition_threshold = 500.0
        self._force_steps_left = 0
        self._enforcement_steps = 1000  # количество шагов принудительного режима

        # Режим фиксированных коэффициентов (для тестов)
        if fixed_kp is not None and fixed_ki is not None and fixed_kd is not None:
            self._fixed_action = (float(fixed_kp), float(fixed_ki), float(fixed_kd))
        else:
            self._fixed_action = None

    def _apply_control_limit_rule(self, control_output: float):
        if control_output < self._force_condition_threshold:
            # Запускаем принудительный режим на полное число шагов,
            # начиная со следующего шага
            self._force_steps_left = self._enforcement_steps
        elif self._force_steps_left > 0:
            # Если режим уже активен и триггер не сработал на этом шаге,
            # уменьшаем оставшееся число шагов
            self._force_steps_left -= 1

    def _get_control_limits_for_step(self) -> tuple[float, float]:
        if self._force_steps_left > 0:
            return self._force_min_value, self._current_max
        return self._current_min, self._current_max

    def _step(self, tensordict: TensorDict) -> TensorDict:
        kp, ki, kd = tensordict["action"].tolist()
        if self._fixed_action is not None:
            kp, ki, kd = self._fixed_action
        applied_min, applied_max = self._get_control_limits_for_step()

        process_variable, control_output, setpoint = self.experimental_setup.step(kp, ki, kd, applied_min, applied_max)

        self._apply_control_limit_rule(control_output)

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
        self._current_min = self._default_min
        self._current_max = self._default_max
        self._force_steps_left = 0

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
    