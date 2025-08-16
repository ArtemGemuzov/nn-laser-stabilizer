import random
from typing import Optional, Tuple

from damped_oscillator import DampedOscillator
from pid_controller import PIDController

from pid_tuning_experimental_setup import PidTuningExperimentalSetup

class OscillatorExperimentalSetup(PidTuningExperimentalSetup):
    def __init__(self, dt: float = 0.01, setpoint: float = 1.0):
        self.system = DampedOscillator(dt=dt)
        self.controller = PIDController(setpoint=setpoint, dt=dt)
        self.setpoint = setpoint
        self.dt = dt
        self.rng = random.Random()

    def step(self, kp: float, ki: float, kd: float) -> Tuple[float, float, float]:
        self.controller.set_params(kp, ki, kd)

        process_variable = self.system.x

        control = self.controller.compute(process_variable)

        new_state = self.system.step(control)

        return new_state, control, self.setpoint

    def reset(self) -> Tuple[float, float, float]:
        self.system.reset()
        self.controller.reset()
        return 0.0, 0.0, self.setpoint

    def set_seed(self, seed: Optional[int]):
        self.rng.seed(seed)