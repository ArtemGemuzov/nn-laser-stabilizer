import time
import random
from typing import Optional

from nn_laser_stabilizer.envs.constants import ADC_MAX, DAC_MAX

class MockSerialConnection:
    def __init__(self, port: str, timeout: float = 0.1, baudrate: int = 115200):
        self.port = port
        self.timeout = timeout
        self.baudrate = baudrate
        self.is_connected = False

        self._step = 0

    def open_connection(self):
        self.is_connected = True
        print("[MOCK_SERIAL_CONNECTION] Serial connection established.")

    def close_connection(self):
        self.is_connected = False
        print("[MOCK_SERIAL_CONNECTION] Serial connection closed.")

    def read_data(self) -> Optional[str]:
        if not self.is_connected:
            raise ConnectionError("[MOCK_SERIAL_CONNECTION] Serial connection is not open.")
        
        process_variable = random.randint(0, ADC_MAX)
        control_output = random.randint(0, DAC_MAX) if self._step < 100 else random.randint(501, DAC_MAX)
        response = f"{process_variable} {control_output}"
        self._step += 1

        print(f"[MOCK_SERIAL_CONNECTION] Read: '{response}' Step: '{self._step}'")
        time.sleep(max(0.001, self.timeout * 0.1))
        return response

    def send_data(self, data_to_send: str):
        if not self.is_connected:
            raise ConnectionError("[MOCK_SERIAL_CONNECTION] Serial connection is not open.")
        print(f"[MOCK_SERIAL_CONNECTION] Send: '{data_to_send}' Step: '{self._step}'")
