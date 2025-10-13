from typing import Optional
import random
import serial

from nn_laser_stabilizer.connection.base_connection import BaseConnection
from nn_laser_stabilizer.envs.constants import ADC_MAX, DAC_MAX

class MockSerialConnection(BaseConnection):
    def __init__(self,
                 port: str,
                 timeout: float = 0.1,
                 baudrate: int = 115200,
                 bytesize: int = serial.EIGHTBITS,
                 parity: str = serial.PARITY_NONE,
                 stopbits: int = serial.STOPBITS_ONE):
        self.port = port  # используется как имя файла для записи
        self.timeout = timeout
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.is_connected = False
        self._log_file = None

    def open_connection(self):
        self.is_connected = True
        try:
            self._log_file = open(self.port, 'a', encoding='utf-8')
            print(f"Mock serial connection established. Logging to: {self.port}")
        except Exception as ex:
            raise ConnectionError(f"Failed to open log file: {self.port}") from ex

    def close_connection(self):
        self.is_connected = False
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        print("Mock serial connection closed.")

    def read_data(self) -> Optional[str]:
        if not self.is_connected:
            raise ConnectionError("Mock serial connection is not open.")
        
        process_variable = random.randint(0, ADC_MAX)
        control_output = random.randint(0, DAC_MAX)
        return f"{process_variable} {control_output}"

    def send_data(self, data_to_send: str):
        if not self.is_connected:
            raise ConnectionError("Mock serial connection is not open.")

        if self._log_file:
            self._log_file.write(data_to_send)
            if not data_to_send.endswith('\n'):
                self._log_file.write('\n')
            self._log_file.flush()

