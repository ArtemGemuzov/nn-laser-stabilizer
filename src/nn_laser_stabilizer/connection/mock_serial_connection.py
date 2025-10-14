from typing import Optional
import random
import os
import serial

from nn_laser_stabilizer.connection import BaseConnection
from nn_laser_stabilizer.envs.constants import ADC_MAX, DAC_MAX
from nn_laser_stabilizer.config.paths import get_hydra_runtime_output_dir

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
        self._read_count = 0
        self._send_count = 0

    def open_connection(self):
        self.is_connected = True
        try:
            log_filename = f"mock_{self.port}.log"
            output_dir = get_hydra_runtime_output_dir()
            log_path = os.path.join(output_dir, log_filename)
            self._log_file = open(log_path, 'a', encoding='utf-8')
            print(f"Mock serial connection established. Logging to: {log_path}")
        except Exception as ex:
            raise ConnectionError(f"Failed to open log file") from ex

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
        data = f"{process_variable} {control_output}\n"
        
        self._read_count += 1
        self._log_file.write(f"#{self._read_count} << {repr(data)}\n")
        self._log_file.flush()
        
        return data

    def send_data(self, data_to_send: str):
        if not self.is_connected:
            raise ConnectionError("Mock serial connection is not open.")

        self._send_count += 1
        self._log_file.write(f"#{self._send_count} >> {repr(data_to_send)}\n")
        self._log_file.flush()

