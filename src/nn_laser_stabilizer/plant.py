from typing import Tuple

import numpy as np

from nn_laser_stabilizer.pid import BaseConnectionToPid


class Plant:
    def __init__(
        self,
        pid_connection: BaseConnectionToPid,
        setpoint: float,
        warmup_steps: int = 1000,
        block_size: int = 100,
        burn_in_steps: int = 20,
        control_output_min_threshold: float = 200.0,
        control_output_max_threshold: float = 4096.0,
        force_min_value: int = 2000,
        force_max_value: int = 2500,
        default_min: int = 0,
        default_max: int = 4095,
        kp_min: float = 2.5,
        kp_max: float = 12.5,
        kp_start: float = 7.5,
        ki_min: float = 0.0,
        ki_max: float = 20.0,
        ki_start: float = 10.0,
        kd_min: float = 0.0,
        kd_max: float = 0.0,
        kd_start: float = 0.0,
    ):
        self.pid_connection = pid_connection
        self._connection_opened = False
        self._setpoint = setpoint

        self._kp_min = kp_min
        self._kp_max = kp_max
        self._kp_start = kp_start
        self._ki_min = ki_min
        self._ki_max = ki_max
        self._ki_start = ki_start
        self._kd_min = kd_min
        self._kd_max = kd_max
        self._kd_start = kd_start

        self._kp = kp_start
        self._ki = ki_start
        self._kd = kd_start
        
        self._warmup_steps = warmup_steps
        self._block_size = block_size
        
        if burn_in_steps >= block_size:
            raise ValueError(
                f"burn_in_steps ({burn_in_steps}) must be less than "
                f"block_size={block_size}, otherwise no data will remain after burn_in"
            )
        
        self._burn_in_steps = burn_in_steps
        
        self._control_output_min_threshold = control_output_min_threshold
        self._control_output_max_threshold = control_output_max_threshold
        
        self._force_min_value = force_min_value
        self._force_max_value = force_max_value
        self._default_min = default_min
        self._default_max = default_max
        
        self._process_variables = np.zeros(block_size, dtype=np.float32)
        self._control_outputs = np.zeros(block_size, dtype=np.float32)
        self._current_index = 0
    
    @property
    def kp(self) -> float:
        return self._kp
    
    @property
    def ki(self) -> float:
        return self._ki
    
    @property
    def kd(self) -> float:
        return self._kd
    
    def update_pid(self, delta_kp: float, delta_ki: float, delta_kd: float) -> None:
        self._kp = np.clip(self._kp + delta_kp, self._kp_min, self._kp_max)
        self._ki = np.clip(self._ki + delta_ki, self._ki_min, self._ki_max)
        self._kd = np.clip(self._kd + delta_kd, self._kd_min, self._kd_max)
    
    def reset_pid(self) -> None:
        self._kp = self._kp_start
        self._ki = self._ki_start
        self._kd = self._kd_start
    
    def _should_reset(self, control_outputs: np.ndarray) -> bool:
        mean_control_output = np.mean(control_outputs)
        return (mean_control_output < self._control_output_min_threshold or 
                mean_control_output > self._control_output_max_threshold)
    
    def step(self) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        self._reset_buffer()
        
        for _ in range(self._block_size):
            process_variable, control_output = self.pid_connection.exchange(
                kp=self._kp,
                ki=self._ki,
                kd=self._kd,
                control_min=self._default_min,
                control_max=self._default_max,
            )
            
            self._process_variables[self._current_index] = process_variable
            self._control_outputs[self._current_index] = control_output
            self._current_index += 1
        
        process_variables, control_outputs = self._get_buffer()
        should_reset = self._should_reset(control_outputs)
        
        return process_variables, control_outputs, self._setpoint, should_reset
    
    def _get_buffer(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self._process_variables[self._burn_in_steps:self._current_index],
            self._control_outputs[self._burn_in_steps:self._current_index]
        )
    
    def _reset_buffer(self) -> None:
        self._current_index = 0
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray, float]:
        if not self._connection_opened:
            self.pid_connection.open()
            self._connection_opened = True
        
        for _ in range(self._warmup_steps):
            self.pid_connection.exchange(
                kp=self._kp,
                ki=self._ki,
                kd=self._kd,
                control_min=self._force_min_value,
                control_max=self._force_max_value,
            )
        
        return self.step()
    
    def close(self) -> None:
        self.pid_connection.close()

