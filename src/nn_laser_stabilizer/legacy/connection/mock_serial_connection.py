from typing import Optional
import random
import os
import math
import serial

from nn_laser_stabilizer.connection import BaseConnection
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
        self._kp: Optional[float] = None
        self._ki: Optional[float] = None
        self._kd: Optional[float] = None
        self._control_min: Optional[float] = None
        self._control_max: Optional[float] = None
        self._setpoint = 1200

    def open(self):
        self.is_connected = True
        try:
            log_filename = f"mock_{self.port}.log"
            output_dir = get_hydra_runtime_output_dir()
            log_path = os.path.join(output_dir, log_filename)
            self._log_file = open(log_path, 'a', encoding='utf-8')
            print(f"Mock serial connection established. Logging to: {log_path}")
        except Exception as ex:
            raise ConnectionError(f"Failed to open log file") from ex

    def close(self):
        self.is_connected = False
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        print("Mock serial connection closed.")

    def read(self) -> Optional[str]:
        if not self.is_connected:
            raise ConnectionError("Mock serial connection is not open.")
        
        kp = self._kp 
        ki = self._ki 
        
        distance = math.sqrt((kp - 10.0) ** 2 + (ki - 17.5) ** 2)
        max_distance = 20.0
        closeness = max(0.0, 1.0 - min(distance, max_distance) / max_distance)

        random_component = random.randint(0, 2000)
        noise = random.gauss(0, 30)

        process_variable = closeness * self._setpoint + (1.0 - closeness) * random_component + noise
        process_variable = int(max(0, min(2000, round(process_variable))))

        control_min = self._control_min
        control_max = self._control_max
        control_output = random.randint(int(control_min), int(control_max))
        data = f"{process_variable} {control_output}\n"
        
        self._read_count += 1
        self._log_file.write(f"#{self._read_count} << {repr(data)}\n")
        self._log_file.flush()
        
        return data

    def send(self, data_to_send: str):
        if not self.is_connected:
            raise ConnectionError("Mock serial connection is not open.")

        self._send_count += 1
        self._log_file.write(f"#{self._send_count} >> {repr(data_to_send)}\n")
        self._log_file.flush()
        try:
            parts = data_to_send.strip().split()
            if len(parts) == 5:
                self._kp = float(parts[0])
                self._ki = float(parts[1])
                self._kd = float(parts[2])
                self._control_min = float(parts[3])
                self._control_max = float(parts[4])
            else:
                raise ValueError(f"Invalid command: '{data_to_send}'")
        except Exception as ex:
            print(f"Failed to parse command: {ex}")

