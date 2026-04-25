from typing import Optional, TypeGuard

import serial


class COMPort:
    def __init__(self,
                 port: str,
                 timeout: float = 0.1,
                 baudrate: int = 115200,
                 bytesize: int = serial.EIGHTBITS,
                 parity: str = serial.PARITY_NONE,
                 stopbits: int = serial.STOPBITS_ONE):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        
        self._serial : Optional[serial.Serial] = None

    def open(self):
        if self._check_connected(self._serial):
            return
        
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=self.bytesize,
                parity=self.parity,
                stopbits=self.stopbits
            )
            if not self._serial.is_open:
                raise ConnectionError("Failed to open serial port.")
        except Exception as ex:
            raise ConnectionError("Error initializing serial connection") from ex

    def close(self):
        if self._check_connected(self._serial):
           self._serial.close()
        self._serial = None

    def read(self) -> str:
        if not self._check_connected(self._serial):
            raise ConnectionError("Serial connection is not open.")

        try:
            raw_data = self._serial.readline()
        except Exception as e:
            raise ConnectionError("Error reading from serial port.") from e

        try:
            data = raw_data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Failed to decode data from serial port: {e}. Raw data: {raw_data!r}."
            ) from e

        if not data:
            raise ValueError("Received empty decoded data from serial port.")
        return data
    
    def send(self, data : str):
        if not self._check_connected(self._serial):
            raise ConnectionError("Serial connection is not open.")
    
        self._serial.write(data.encode('utf-8'))

    @staticmethod
    def _check_connected(value: Optional[serial.Serial]) -> TypeGuard[serial.Serial]:
        return value is not None and value.is_open