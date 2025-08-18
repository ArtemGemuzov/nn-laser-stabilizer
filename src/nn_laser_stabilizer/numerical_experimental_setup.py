from typing import Optional, Tuple

from damped_oscillator import DampedOscillator
from pid_controller import PIDController

from pid_tuning_experimental_setup import PidTuningExperimentalSetup

class NumericalExperimentalSetup(PidTuningExperimentalSetup):
    def __init__(self, system : DampedOscillator, controller : PIDController, dt : float = 0.01):
        self.system = system
        self.controller = controller

        self.dt = dt

    def step(self, kp: float, ki: float, kd: float) -> Tuple[float, float, float]:
        self.controller.set_params(kp, ki, kd)

        process_variable = self.system.x

        control = self.controller(process_variable, self.dt)
        new_state = self.system.step(control, self.dt)

        return new_state, control, self.controller.setpoint

    def reset(self) -> Tuple[float, float, float]:
        process_variable = self.system.reset()
        self.controller.reset()

        return self.system.x, 0.0, self.controller.setpoint

    def set_seed(self, seed: Optional[int]):
        self.system.set_seed(seed)