import time
from typing import Optional

class MockSerialConnection:
    def __init__(self, port: str, timeout: float = 0.1, baudrate: int = 115200):
        self.port = port
        self.timeout = timeout
        self.baudrate = baudrate

        self.is_connected = False

        self.process_variable = 1.0
        self.control_output = 1.0

    def open_connection(self):
        self.is_connected = True
        print("[MOCK_SERIAL_CONNECTION] Serial connection established.")

    def close_connection(self):
        self.is_connected = False
        print("[MOCK_SERIAL_CONNECTION] Serial connection closed.")

    def read_data(self) -> Optional[str]:
        if not self.is_connected:
            raise ConnectionError("[MOCK_SERIAL_CONNECTION] Serial connection is not open.")
        
        response = f"{int(self.process_variable)} {int(self.control_output)}"
        print(f"[MOCK_SERIAL_CONNECTION] Read: '{response}'")
        time.sleep(max(0.001, self.timeout * 0.1))
        return response

    def send_data(self, data_to_send: str):
        if not self.is_connected:
            raise ConnectionError("[MOCK_SERIAL_CONNECTION] Serial connection is not open.")
        
        print(f"[MOCK_SERIAL_CONNECTION] Send: '{repr(data_to_send)}'")
