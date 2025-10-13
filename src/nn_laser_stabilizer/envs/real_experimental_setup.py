from typing import Tuple, Optional

from nn_laser_stabilizer.envs.pid_tuning_experimental_setup import PidTuningExperimentalSetup
from nn_laser_stabilizer.envs.constants import (
    DEFAULT_KP,
    DEFAULT_KI,
    DEFAULT_KD,
    DEFAULT_MIN_CONTROL,
    DEFAULT_MAX_CONTROL,
)
from nn_laser_stabilizer.connection import ConnectionToPid

class RealExperimentalSetup(PidTuningExperimentalSetup):
    """
    Реализация протокола PidTuningExperimentalSetup для реальной установки через SerialConnection.
    Формат обмена:
        → Команда: "kp ki kd u_min u_max"
        ← Ответ: "PV CO"
    """
    def __init__(self, serial_connection, setpoint: float):
        self.pid_connection = ConnectionToPid(serial_connection)
        self.setpoint = setpoint

    def step(self, kp: float, ki: float, kd: float, control_min: float, control_max: float) -> Tuple[float, float, float]:
        self.pid_connection.send_pid_command(
            kp=kp,
            ki=ki,
            kd=kd,
            control_min=control_min,
            control_max=control_max,
        )
        process_variable, control_output = self.pid_connection.read_data_and_wait()
        return process_variable, control_output, self.setpoint
            
    def reset(self) -> Tuple[float, float, float]:
        # TODO: уточнить, должны ли мы инициировать обмен
        self.pid_connection.send_pid_command(
            kp=DEFAULT_KP,
            ki=DEFAULT_KI,
            kd=DEFAULT_KD,
            control_min=DEFAULT_MIN_CONTROL,
            control_max=DEFAULT_MAX_CONTROL,
        )
        process_variable, control_output = self.pid_connection.read_data_and_wait()
        return process_variable, control_output, self.setpoint

    def set_seed(self, seed: Optional[int]):
        pass