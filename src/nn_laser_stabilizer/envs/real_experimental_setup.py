from typing import Tuple, Optional

from nn_laser_stabilizer.envs.pid_tuning_experimental_setup import PidTuningExperimentalSetup
from nn_laser_stabilizer.envs.constants import DAC_MAX
from nn_laser_stabilizer.connection.pid_serial_connection import PidSerialConnection

class RealExperimentalSetup(PidTuningExperimentalSetup):
    """
    Реализация протокола PidTuningExperimentalSetup для реальной установки через SerialConnection.
    Формат обмена:
        → Команда: "kp ki kd u_min u_max"
        ← Ответ: "PV CO"
    """

    DEFAULT_KP = 3.5
    DEFAULT_KI = 11.0
    DEFAULT_KD = 0.002

    DEFAULT_MIN_CONTROL = 0
    DEFAULT_MAX_CONTROL = DAC_MAX

    def __init__(self, serial_connection, setpoint: float):
        self.pid_connection = PidSerialConnection(serial_connection)
        self.setpoint = setpoint

    def step(self, kp: float, ki: float, kd: float, control_min: float, control_max: float) -> Tuple[float, float, float]:
        self.pid_connection.send_pid_command(
            kp=kp,
            ki=ki,
            kd=kd,
            control_min=control_min,
            control_max=control_max,
        )

        while True:
            response = self.pid_connection.read_data()
            if response:
                process_variable, control_output = response
                return process_variable, control_output, self.setpoint
            
    def reset(self) -> Tuple[float, float, float]:
        self.pid_connection.send_pid_command(
            kp=self.DEFAULT_KP,
            ki=self.DEFAULT_KI,
            kd=self.DEFAULT_KD,
            control_min=self.DEFAULT_MIN_CONTROL,
            control_max=self.DEFAULT_MAX_CONTROL,
        )
        while True:
            response = self.pid_connection.read_data()
            if response:
                process_variable, control_output = response
                return process_variable, control_output, self.setpoint

    def set_seed(self, seed: Optional[int]):
        pass