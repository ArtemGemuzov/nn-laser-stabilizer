import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import UnboundedContinuous, BoundedContinuous

from nn_laser_stabilizer.pid_tuning_experimental_setup import PidTuningExperimentalSetup

class PidTuningExperimentalEnv(EnvBase):
    def __init__(self, experimental_setup : PidTuningExperimentalSetup):
        super().__init__()

        self.experimental_setup = experimental_setup

        # Действия: [Kp, Ki, Kd]
        self.action_spec = BoundedContinuous(low=0, high=100, shape=(3,))
        # Наблюдения: [process_variable, control_output, setpoint]
        self.observation_spec = UnboundedContinuous(shape=(3,))
        # Вознаграждение: скаляр
        self.reward_spec = UnboundedContinuous(shape=(1,))

    def _step(self, tensordict: TensorDict) -> TensorDict:
        kp, ki, kd = tensordict["action"].tolist()

        process_variable, control_output, setpoint = self.experimental_setup.step(kp, ki, kd)

        observation = torch.tensor(
            [process_variable, control_output, setpoint],
            dtype=torch.float32,
            device=self.device
        )

        error = setpoint - process_variable
        reward = -abs(error)

        done = False  # задать условие завершения

        return TensorDict(
            {
                "observation": observation,
                "reward": torch.tensor([reward], device=self.device),
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
            },
            batch_size=[]
        )

    def _reset(self, unused: TensorDict | None = None) -> TensorDict:
        process_variable, control_output, setpoint = self.experimental_setup.reset()

        observation = torch.tensor(
            [process_variable, control_output, setpoint],
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