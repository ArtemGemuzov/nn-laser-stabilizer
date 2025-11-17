from typing import Tuple, Optional
import numpy as np

from nn_laser_stabilizer.envs.experimental_setup_protocol import ExperimentalSetupProtocol
from nn_laser_stabilizer.envs.simulation.pid_controller import PIDController


class NumericalExperimentalSetupController(ExperimentalSetupProtocol): 
    def __init__(
        self,
        plant,
        setpoint: float = 1.0,
        dt: float = 0.01,
        warmup_steps: int = 200,
        block_size: int = 100,
    ):
        self._setpoint = float(setpoint)
        self._dt = float(dt)
        self._warmup_steps = int(warmup_steps)
        self._block_size = int(block_size)

        self._plant = plant
        self._pid = PIDController(setpoint=self._setpoint)

        buffer_size = max(self._warmup_steps, self._block_size)
        self._process_variables = np.zeros(buffer_size, dtype=np.float32)
        self._control_outputs = np.zeros(buffer_size, dtype=np.float32)
        self._setpoints = np.zeros(buffer_size, dtype=np.float32)
        self._current_index = 0

    @property
    def setpoint(self) -> float:
        return self._setpoint

    def _reset_buffer(self) -> None:
        self._current_index = 0

    def _append_sample(self, process_variable: float, control_output: float) -> None:
        self._process_variables[self._current_index] = process_variable
        self._control_outputs[self._current_index] = control_output
        self._setpoints[self._current_index] = self.setpoint
        self._current_index += 1

    def _get_buffer(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self._process_variables[:self._current_index],
            self._control_outputs[:self._current_index],
            self._setpoints[:self._current_index],
        )

    def step(self, kp: float, ki: float, kd: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._reset_buffer()
        self._pid.set_params(kp, ki, kd)

        for _ in range(self._block_size):
            pv = self._plant.process_variable
            control = self._pid(pv, dt=self._dt)
            pv_next = self._plant.step(control, dt=self._dt)
            self._append_sample(pv_next, control)

        return self._get_buffer()
    
    def reset(self, kp: float, ki: float, kd: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._reset_buffer()

        self._pid.reset()
        self._pid.set_params(kp, ki, kd)
        self._plant.reset()

        for _ in range(self._warmup_steps):
            pv = self._plant.process_variable
            control = self._pid(pv, dt=self._dt)
            pv_next = self._plant.step(control, dt=self._dt)
            self._append_sample(pv_next, control)

        return self._get_buffer()
    
    def set_seed(self, seed: Optional[int]) -> None:
        pass
    
    def close(self) -> None:
        pass
