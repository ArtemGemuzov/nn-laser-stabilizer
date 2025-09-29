import time
import random
from typing import Optional

from nn_laser_stabilizer.envs.constants import ADC_MAX, DAC_MAX, DEFAULT_KP, DEFAULT_KI, DEFAULT_KD, KP_MIN, KP_MAX, KI_MIN, KI_MAX, KD_MAX, KD_MIN
from nn_laser_stabilizer.connection.base_connection import BaseConnection

class MockSerialConnection(BaseConnection):
    def __init__(self, port: str, timeout: float = 0.1, baudrate: int = 115200):
        self.port = port
        self.timeout = timeout
        self.baudrate = baudrate
        self.is_connected = False

        self._step = 0
        self.setpoint = 1200 
        self.current_kp = DEFAULT_KP
        self.current_ki = DEFAULT_KI
        self.current_kd = DEFAULT_KD

    def open_connection(self):
        self.is_connected = True
        print("[MOCK_SERIAL_CONNECTION] Serial connection established.")

    def close_connection(self):
        self.is_connected = False
        print("[MOCK_SERIAL_CONNECTION] Serial connection closed.")

    def read_data(self) -> Optional[str]:
        if not self.is_connected:
            raise ConnectionError("[MOCK_SERIAL_CONNECTION] Serial connection is not open.")

        eff_kp = 1 - abs(self.current_kp - DEFAULT_KP) / (KP_MAX - KP_MIN)
        eff_ki = 1 - abs(self.current_ki - DEFAULT_KI) / (KI_MAX - KI_MIN)
        eff_kd = 1 - abs(self.current_kd - DEFAULT_KD) / (KD_MAX - KD_MIN)
        efficiency = max(0, min(1, (eff_kp + eff_ki + eff_kd) / 3))

        base_process = self.setpoint + (1 - efficiency) * (ADC_MAX // 2)
        noise = random.randint(-20, 20)  
        process_variable = int(base_process + noise)
    
        control_output = min(max(0, int((self.setpoint - process_variable) * 0.5 + random.randint(-5, 5))), DAC_MAX)

        self._step += 1
        response = f"{process_variable} {control_output}"

        print(f"[MOCK_SERIAL_CONNECTION] Read: '{response}' Step: {self._step} Efficiency: {efficiency:.3f}")
        return response

    def send_data(self, data_to_send: str):
        if not self.is_connected:
            raise ConnectionError("[MOCK_SERIAL_CONNECTION] Serial connection is not open.")

        try:
            kp_str, ki_str, kd_str, _, _ = data_to_send.strip().split()
            self.current_kp = float(kp_str)
            self.current_ki = float(ki_str)
            self.current_kd = float(kd_str)
        except Exception:
            pass

        print(f"[MOCK_SERIAL_CONNECTION] Send: '{data_to_send}' Step: {self._step}")

