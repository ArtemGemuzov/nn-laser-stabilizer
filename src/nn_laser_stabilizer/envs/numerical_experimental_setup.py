from typing import Optional, Tuple

from nn_laser_stabilizer.envs.oscillator import DuffingOscillator
from nn_laser_stabilizer.envs.pid_controller import PIDController
from nn_laser_stabilizer.envs.pid_tuning_experimental_setup import PidTuningExperimentalSetup

class NumericalExperimentalSetup(PidTuningExperimentalSetup):
    def __init__(self, system : DuffingOscillator, controller : PIDController, dt : float = 0.01):
        self.system = system
        self.controller = controller

        self.dt = dt

    def step(self, kp: float, ki: float, kd: float, control_min: float, control_max: float) -> Tuple[float, float, float]:
        # Игнорирует параметры control_min и control_max
        self.controller.set_params(kp, ki, kd)

        control = self.controller(self.system.process_variable, self.dt)
        new_state = self.system.step(control, self.dt)

        return new_state, control, self.controller.setpoint

    def reset(self) -> Tuple[float, float, float]:
        self.controller.reset()

        return self.system.process_variable, 0.0, self.controller.setpoint

    def set_seed(self, seed: Optional[int]):
        self.system.set_seed(seed)