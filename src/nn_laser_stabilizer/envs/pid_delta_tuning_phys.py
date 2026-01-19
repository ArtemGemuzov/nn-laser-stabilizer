from typing import Tuple

import numpy as np

from nn_laser_stabilizer.hardware.connection import create_connection
from nn_laser_stabilizer.connection.pid_connection import (
    ConnectionToPid,
    LoggingConnectionToPid,
    ConnectionToPidProtocol,
)
from nn_laser_stabilizer.connection.pid_protocol import PidProtocol
from nn_laser_stabilizer.logger import Logger, PrefixedLogger


def determine_setpoint(
    pid_connection: ConnectionToPidProtocol,
    steps: int,
    max_value: int,
    factor: float,
) -> tuple[int, int, int]:
    if steps <= 1:
        raise ValueError(f"steps ({steps}) must be greater than 1")
    
    min_pv = float('inf')
    max_pv = float('-inf')
    
    for step in range(steps):
        progress = step / (steps - 1)
        control_min = int(progress * max_value)
        control_max = int(progress * max_value)
        
        process_variable, _ = pid_connection.exchange(
            kp=0.0,
            ki=0.0,
            kd=0.0,
            control_min=control_min,
            control_max=control_max,
            setpoint=0,
        )
        
        min_pv = min(min_pv, process_variable)
        max_pv = max(max_pv, process_variable)
    
    min_pv_int = int(min_pv)
    max_pv_int = int(max_pv)
    setpoint = int(round(min_pv + factor * (max_pv - min_pv)))
    return setpoint, min_pv_int, max_pv_int


class PidDeltaTuningPhys:
    LOG_PREFIX = "PLANT"
    
    def __init__(
        self,
        *,
        # Логгер верхнего уровня
        logger: Logger,
        # Параметры для соединения
        port: str,
        timeout: float,
        baudrate: int,
        # Параметры для логирования соединения
        log_connection: bool,
        # Параметры для работы с установкой
        setpoint: int,
        warmup_steps: int,
        block_size: int,
        burn_in_steps: int,
        control_output_min_threshold: float,
        control_output_max_threshold: float,
        force_min_value: int,
        force_max_value: int,
        default_min: int,
        default_max: int,
        kp_min: float,
        kp_max: float,
        kp_start: float,
        ki_min: float,
        ki_max: float,
        ki_start: float,
        kd_min: float,
        kd_max: float,
        kd_start: float,
        auto_determine_setpoint: bool,
        setpoint_determination_steps: int,
        setpoint_determination_max_value: int,
        setpoint_determination_factor: float,
    ):
        connection = create_connection(
            port=port,
            timeout=timeout,
            baudrate=baudrate,
        )
        pid_connection: ConnectionToPid | LoggingConnectionToPid = ConnectionToPid(
            connection=connection
        )

        if log_connection:
            pid_connection = LoggingConnectionToPid(
                connection_to_pid=pid_connection,
                logger=logger,
            )

        self.pid_connection = pid_connection
        
        self._logger = PrefixedLogger(logger, PidDeltaTuningPhys.LOG_PREFIX)

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
        
        self._setpoint = setpoint
        self._auto_determine_setpoint = auto_determine_setpoint
        self._setpoint_determination_steps = setpoint_determination_steps
        self._setpoint_determination_max_value = setpoint_determination_max_value
        self._setpoint_determination_factor = setpoint_determination_factor
        self._setpoint_determined = False
        
        if auto_determine_setpoint and setpoint_determination_steps <= 1:
            raise ValueError(
                f"setpoint_determination_steps ({setpoint_determination_steps}) must be greater than 1 "
                f"when auto_determine_setpoint is True"
            )
    
    @property
    def setpoint(self) -> int:
        return self._setpoint
    
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
        
        self._kp = round(self._kp, PidProtocol.KP_DECIMAL_PLACES)
        self._ki = round(self._ki, PidProtocol.KI_DECIMAL_PLACES)
        self._kd = round(self._kd, PidProtocol.KD_DECIMAL_PLACES)
    
    def reset_pid(self) -> None:
        self._kp = self._kp_start
        self._ki = self._ki_start
        self._kd = self._kd_start
    
    def _should_reset(self, control_outputs: np.ndarray) -> bool:
        mean_control_output = np.mean(control_outputs)
        return bool(mean_control_output < self._control_output_min_threshold or 
                mean_control_output > self._control_output_max_threshold)
    
    def _determine_setpoint(self) -> None:
        setpoint, min_pv_int, max_pv_int = determine_setpoint(
            pid_connection=self.pid_connection,
            steps=self._setpoint_determination_steps,
            max_value=self._setpoint_determination_max_value,
            factor=self._setpoint_determination_factor,
        )
        
        self._setpoint = setpoint
        self._setpoint_determined = True
        
        self._logger.log(
            f"setpoint determined: setpoint={self._setpoint} "
            f"min_pv={min_pv_int} max_pv={max_pv_int}"
        )
        
        # TODO: Заменить print на ConsoleLogger для унифицированного вывода в консоль
        print(f"Setpoint determined: {self._setpoint} (min_pv={min_pv_int}, max_pv={max_pv_int})")
    
    def step(self) -> Tuple[np.ndarray, np.ndarray, int, bool]:
        self._reset_buffer()
        
        for _ in range(self._block_size):
            process_variable, control_output = self.pid_connection.exchange(
                kp=self._kp,
                ki=self._ki,
                kd=self._kd,
                control_min=self._default_min,
                control_max=self._default_max,
                setpoint=self._setpoint,
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
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray, int, bool]:
        self.pid_connection.open()
        
        if self._auto_determine_setpoint and not self._setpoint_determined:
            self._determine_setpoint()
        
        for _ in range(self._warmup_steps):
            self.pid_connection.exchange(
                kp=self._kp,
                ki=self._ki,
                kd=self._kd,
                control_min=self._force_min_value,
                control_max=self._force_max_value,
                setpoint=self._setpoint,
            )
        
        return self.step()
    
    def close(self) -> None:
        self.pid_connection.close()
