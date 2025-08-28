from typing import Tuple, Optional

from nn_laser_stabilizer.envs.pid_tuning_experimental_setup import PidTuningExperimentalSetup
from nn_laser_stabilizer.serial_connection import SerialConnection
from nn_laser_stabilizer.mock_serial_connection import MockSerialConnection

class RealExperimentalSetup(PidTuningExperimentalSetup):
    """
    Реализация протокола PidTuningExperimentalSetup для реальной установки через SerialConnection.
    Формат обмена:
        → Команда: "kp ki kd"
        ← Ответ: "PV CO"
    """

    DEFAULT_KP = 3.5
    DEFAULT_KI = 11.0
    DEFAULT_KD = 0.002

    def __init__(self, serial_connection, setpoint: float):
        self.serial_connection = serial_connection
        self.setpoint = setpoint

    def _parse_response(self, response: str) -> Tuple[float, float]:
        try:
            parts = response.strip().split()
            if len(parts) != 2:
                raise ValueError(f"Expected 2 values, got {len(parts)}")
            return float(parts[0]), float(parts[1])
        except Exception as ex:
            raise ValueError(f"Invalid response format: '{response}'") from ex

    def step(self, kp: float, ki: float, kd: float) -> Tuple[float, float, float]:
        command = f"{kp:.4f} {ki:.4f} {kd:.4f}\n"
        self.serial_connection.send_data(command)

        while True:
            response = self.serial_connection.read_data()
            if response:
                process_variable, control_output = self._parse_response(response)
                return process_variable, control_output, self.setpoint
            
    def reset(self) -> Tuple[float, float, float]:
        command = f"{self.DEFAULT_KP:.4f} {self.DEFAULT_KI:.4f} {self.DEFAULT_KD:.4f}"
        self.serial_connection.send_data(command)
        while True:
            response = self.serial_connection.read_data()
            if response:
                process_variable, control_output = self._parse_response(response)
                return process_variable, control_output, self.setpoint

    def set_seed(self, seed: Optional[int]):
        pass