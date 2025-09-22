import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase

from nn_laser_stabilizer.envs.pid_tuning_experimental_setup import PidTuningExperimentalSetup
from nn_laser_stabilizer.envs.normalization import normalize_adc, normalize_dac

class PidTuningExperimentalEnv(EnvBase):
    def __init__(self, 
                 experimental_setup : PidTuningExperimentalSetup, 
                 action_spec,
                 observation_spec,
                 reward_spec,
                 reward_func
    ):
        super().__init__()

        self.experimental_setup = experimental_setup

        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec

        self.reward_func = reward_func

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action_values = tensordict["action"].tolist()
        if len(action_values) == 3:
            kp, ki, kd = action_values
            # по умолчанию используем полный диапазон DAC
            from nn_laser_stabilizer.envs.constants import DAC_MAX
            control_min, control_max = 0.0, DAC_MAX
        elif len(action_values) == 5:
            kp, ki, kd, control_min, control_max = action_values
        else:
            raise ValueError(f"Expected action of length 3 or 5, got {len(action_values)}")

        process_variable, control_output, setpoint = self.experimental_setup.step(kp, ki, kd, control_min, control_max)

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
    