from torchrl.envs import TransformedEnv, DoubleToFloat, Compose, InitTracker, GymEnv, StepCounter

from nn_laser_stabilizer.envs.pid_controller import PIDController
from nn_laser_stabilizer.envs.oscillator import DuffingOscillator
from nn_laser_stabilizer.envs.numerical_experimental_setup import NumericalExperimentalSetup
from nn_laser_stabilizer.envs.pid_tuning_experimental_env import PidTuningExperimentalEnv

from torchrl.envs.transforms import Transform, Compose, DoubleToFloat, ObservationNorm, RewardScaling
from torchrl.envs.transforms import StepCounter, InitTracker
from torchrl.data import UnboundedContinuous, BoundedContinuous
import torch

from torch.utils.tensorboard import SummaryWriter

class PerStepLogger(Transform):
    """
    Логирует kp, ki, kd (из action) и x, setpoint (из observation) каждый шаг.
    """
    def __init__(self, writer=None):
        super().__init__()
        self.writer = writer
        self._t = 0 

    def _log_step(self, action_row: torch.Tensor, observation_row: torch.Tensor):
        kp = action_row[0].item()
        ki = action_row[1].item()
        kd = action_row[2].item()

        x = observation_row[0].item()
        setpoint = observation_row[2].item()

        self.writer.add_scalar("Action/kp", kp, self._t)
        self.writer.add_scalar("Action/ki", ki, self._t)
        self.writer.add_scalar("Action/kd", kd, self._t)
        self.writer.add_scalar("Observation/x", x, self._t)
        self.writer.add_scalar("Observation/setpoint", setpoint, self._t)

        self._t += 1

    def _step(self, tensordict, next_tensordict) :
        """
        Вызывается на каждом env.step().
        tensordict: данные на шаге t
        next_tensordict: данные на шаге t+1
        """      
        action = tensordict.get("action", None)
        observation = tensordict.get("observation", None)

        if action is not None and observation is not None:
            self._log_step(action, observation)

        return next_tensordict

def make_specs(bounds_config: dict) -> dict:
    specs = {}
    for key in ["action", "observation", "reward"]:
        spec_bounds = bounds_config.get(key)
        if spec_bounds is None:
            raise ValueError(f"Missing bounds for {key}")

        low = torch.tensor([float(x) for x in spec_bounds["low"]])
        high = torch.tensor([float(x) for x in spec_bounds["high"]])

        if torch.isinf(low).any() or torch.isinf(high).any():
            specs[key] = UnboundedContinuous(shape=low.shape)
        else:
            specs[key] = BoundedContinuous(low=low, high=high, shape=low.shape)

    return specs

def make_env(config) -> TransformedEnv:
    env_config = config.env

    pid = PIDController(setpoint=env_config.setpoint)
    oscillator = DuffingOscillator(
        mass=env_config.mass, 
        k_linear=env_config.k_linear, 
        k_nonlinear=env_config.k_nonlinear, 
        k_damping=env_config.k_damping,
        process_noise_std=env_config.process_noise_std, 
        measurement_noise_std=env_config.measurement_noise_std
    )
    numerical_model = NumericalExperimentalSetup(oscillator, pid)

    specs = make_specs(env_config.bounds)

    base_env = PidTuningExperimentalEnv(
        numerical_model, 
        action_spec=specs["action"],
        observation_spec=specs["observation"],
        reward_spec=specs["reward"]
    )
    
    env = TransformedEnv(
        base_env,
        Compose(
            StepCounter(),
            InitTracker(),
            DoubleToFloat(),
        )
    )
    env.set_seed(config.seed)
    return env

def add_logger_to_env(env: TransformedEnv, logdir) -> TransformedEnv:
    writer = SummaryWriter(log_dir=logdir)
    env.append_transform(PerStepLogger(writer=writer))
    return env