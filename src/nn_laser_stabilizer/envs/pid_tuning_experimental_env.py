import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase

from nn_laser_stabilizer.envs.pid_tuning_experimental_setup import PidTuningExperimentalSetup

class PidTuningExperimentalEnv(EnvBase):
    ADC_MAX = 10230.0 
    DAC_MAX = 4095.0   

    def __init__(self, 
                 experimental_setup : PidTuningExperimentalSetup, 
                 action_spec,
                 observation_spec,
                 reward_spec
    ):
        super().__init__()

        self.experimental_setup = experimental_setup

        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec
    
    def _normalize_adc(self, value: float) -> float:
        return (value / self.ADC_MAX) * 2.0 - 1.0

    def _normalize_dac(self, value: float) -> float:
        return (value / self.DAC_MAX) * 2.0 - 1.0

    def _step(self, tensordict: TensorDict) -> TensorDict:
        kp, ki, kd = tensordict["action"].tolist()

        process_variable, control_output, setpoint = self.experimental_setup.step(kp, ki, kd)

        # лежит в [-1; 1]
        process_variable_norm = self._normalize_adc(process_variable)
        setpoint_norm = self._normalize_adc(setpoint)
        control_output_norm = self._normalize_dac(control_output)

        observation = torch.tensor(
            [process_variable_norm, control_output_norm, setpoint_norm],
            dtype=torch.float32,
            device=self.device
        )

        error = setpoint_norm - process_variable_norm 
        reward = -abs(error) # лежит в [-2; 0]
        reward_norm = 1 + reward

        done = False  # TODO задать условие завершения

        return TensorDict(
            {
                "observation": observation,
                "reward": torch.tensor([reward_norm], dtype=torch.float32, device=self.device),
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
            },
            batch_size=[]
        )

    def _reset(self, unused: TensorDict | None = None) -> TensorDict:
        process_variable, control_output, setpoint = self.experimental_setup.reset()

        process_variable_norm = self._normalize_adc(process_variable)
        setpoint_norm = self._normalize_adc(setpoint)
        control_output_norm = self._normalize_dac(control_output)

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
    