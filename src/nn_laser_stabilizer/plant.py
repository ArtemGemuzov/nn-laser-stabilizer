from typing import Tuple, Optional
import numpy as np

from nn_laser_stabilizer.pid import BaseConnectionToPid


class Plant:
    """
    Управляет экспериментальной системой (plant) через PID соединение.
    Собирает данные процесса и управляющих воздействий.
    """
    
    # Константы для PID параметров
    KP_MIN = 2.5
    KP_MAX = 12.5
    KP_START = 7.5
    
    KI_MIN = 0.0
    KI_MAX = 20.0
    KI_START = 10.0
    
    KD_MIN = 0.0
    KD_MAX = 0.0  # Фиксирован на 0
    KD_START = 0.0
    
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
    ):
        self.pid_connection = pid_connection
        self._connection_opened = False
        self._setpoint = setpoint

        self._kp = self.KP_START
        self._ki = self.KI_START
        self._kd = self.KD_START
        
        self._warmup_steps = warmup_steps
        self._block_size = block_size
        
        min_size = min(warmup_steps, block_size)
        if burn_in_steps >= min_size:
            raise ValueError(
                f"burn_in_steps ({burn_in_steps}) must be less than "
                f"min(warmup_steps={warmup_steps}, block_size={block_size})={min_size}, "
                f"otherwise no data will remain after burn_in"
            )
        
        self._burn_in_steps = burn_in_steps
        
        self._control_output_min_threshold = control_output_min_threshold
        self._control_output_max_threshold = control_output_max_threshold
        
        self._force_min_value = force_min_value
        self._force_max_value = force_max_value
        self._default_min = default_min
        self._default_max = default_max
        
        buffer_size = max(warmup_steps, block_size)
        self._process_variables = np.zeros(buffer_size, dtype=np.float32)
        self._control_outputs = np.zeros(buffer_size, dtype=np.float32)
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
        self._kp = np.clip(self._kp + delta_kp, self.KP_MIN, self.KP_MAX)
        self._ki = np.clip(self._ki + delta_ki, self.KI_MIN, self.KI_MAX)
        self._kd = np.clip(self._kd + delta_kd, self.KD_MIN, self.KD_MAX)
    
    def reset_pid(self) -> None:
        self._kp = self.KP_START
        self._ki = self.KI_START
        self._kd = self.KD_START
    
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
        self._reset_buffer()
        
        if not self._connection_opened:
            self.pid_connection.open_connection()
            self._connection_opened = True
        
        for _ in range(self._warmup_steps):
            process_variable, control_output = self.pid_connection.exchange(
                kp=self._kp,
                ki=self._ki,
                kd=self._kd,
                control_min=self._force_min_value,
                control_max=self._force_max_value,
            )
            
            self._process_variables[self._current_index] = process_variable
            self._control_outputs[self._current_index] = control_output
            self._current_index += 1
        
        process_variables, control_outputs = self._get_buffer()
        return process_variables, control_outputs, self._setpoint
    
    def close(self) -> None:
        self.pid_connection.close_connection()

