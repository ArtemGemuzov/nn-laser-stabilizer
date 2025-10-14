from typing import Tuple, Optional
import numpy as np

from nn_laser_stabilizer.connection import BaseConnectionToPid
from nn_laser_stabilizer.envs.constants import (
    DEFAULT_KP,
    DEFAULT_KI,
    DEFAULT_KD,
)


class ExperimentalSetupController:
    def __init__(
        self,
        pid_connection: BaseConnectionToPid,
        setpoint: float,
        warmup_steps: int = 1000,
        block_size: int = 100,
        max_buffer_size: int = 2000,
        force_min_value: float = 2000.0,
        force_max_value: float = 4095.0,
        default_min: float = 0.0,
        default_max: float = 4095.0,
    ):
        self.pid_connection = pid_connection
        self.setpoint = setpoint
        
        self._warmup_steps = warmup_steps
        self._block_size = block_size
        
        self._force_min_value = force_min_value
        self._force_max_value = force_max_value
        self._default_min = default_min
        self._default_max = default_max
        
        self._buffer_size = max_buffer_size
        self._process_variables = np.zeros(max_buffer_size, dtype=np.float32)
        self._control_outputs = np.zeros(max_buffer_size, dtype=np.float32)
        self._setpoints = np.zeros(max_buffer_size, dtype=np.float32)
        self._current_index = 0
    
    def step(self, kp: float, ki: float, kd: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._reset_buffer()
        
        for _ in range(self._block_size):
            process_variable, control_output = self.pid_connection.exchange(
                kp=kp,
                ki=ki,
                kd=kd,
                control_min=self._default_min,
                control_max=self._default_max,
            )
            
            self._process_variables[self._current_index] = process_variable
            self._control_outputs[self._current_index] = control_output
            self._setpoints[self._current_index] = self.setpoint
            self._current_index += 1
        
        return self._get_buffer()
    
    def _get_buffer(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self._process_variables[:self._current_index],
            self._control_outputs[:self._current_index],
            self._setpoints[:self._current_index]
        )
    
    def _reset_buffer(self) -> None:
        self._current_index = 0
    
    def reset(self, kp: float = DEFAULT_KP, ki: float = DEFAULT_KI, kd: float = DEFAULT_KD) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._reset_buffer()
        
        self.pid_connection.open_connection()
        for _ in range(self._warmup_steps):
            process_variable, control_output = self.pid_connection.exchange(
                kp=kp,
                ki=ki,
                kd=kd,
                control_min=self._force_min_value,
                control_max=self._force_max_value,
            )
            
            self._process_variables[self._current_index] = process_variable
            self._control_outputs[self._current_index] = control_output
            self._setpoints[self._current_index] = self.setpoint
            self._current_index += 1
        
        return self._get_buffer()
    
    def close(self) -> None:
        """Закрывает соединение с установкой."""
        self.pid_connection.close_connection()
    
    def set_seed(self, seed: Optional[int]) -> None:
        """Для реальной установки seed не применяется."""
        pass


