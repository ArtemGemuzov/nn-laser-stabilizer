from typing import Tuple, Optional

from nn_laser_stabilizer.envs.pid_tuning_experimental_setup import PidTuningExperimentalSetup
from nn_laser_stabilizer.serial_connection import SerialConnection

class RealExperimentalSetup(PidTuningExperimentalSetup):
    """
    Реализация протокола PidTuningExperimentalSetup для реальной установки через SerialConnection.
    Формат обмена:
        → Команда: "kp ki kd"
        ← Ответ: "PV CO SP"
    """

    DEFAULT_KP = 3.5
    DEFAULT_KI = 11.0
    DEFAULT_KD = 0.002

    def __init__(self, serial_connection: SerialConnection):
        self.serial_connection = serial_connection

    def _parse_response(self, response: str) -> Tuple[float, float, float]:
        try:
            parts = response.strip().split()
            if len(parts) != 3:
                raise ValueError(f"Expected 3 values, got {len(parts)}")
            return float(parts[0]), float(parts[1]), float(parts[2])
        except Exception as ex:
            raise ValueError(f"Invalid response format: '{response}'") from ex

    def step(self, kp: float, ki: float, kd: float) -> Tuple[float, float, float]:
        command = f"{kp:.4f} {ki:.4f} {kd:.4f}"
        self.serial_connection.send_data(command)

        while True:
            response = self.serial_connection.read_data()
            if response:
                return self._parse_response(response)

    def reset(self) -> Tuple[float, float, float]:
        command = f"{self.DEFAULT_KP:.4f} {self.DEFAULT_KI:.4f} {self.DEFAULT_KD:.4f}"
        self.serial_connection.send_data(command)
        while True:
            response = self.serial_connection.read_data()
            if response:
                return self._parse_response(response)

    def set_seed(self, seed: Optional[int]):
        pass