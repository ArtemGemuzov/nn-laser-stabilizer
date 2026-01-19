from typing import Optional, TypeGuard
import warnings

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
        
        while True:
            try:
                raw_data = self._serial.readline()
                if not raw_data:
                    continue
                
                try:
                    data = raw_data.decode("utf-8")
                    if data:
                        return data
                except UnicodeDecodeError as e:
                    warnings.warn(
                        f"Failed to decode data from serial port: {e}. "
                        f"Raw data: {raw_data!r}. Retrying...",
                        RuntimeWarning,
                        stacklevel=2
                    )
                    continue
            except Exception as e:
                warnings.warn(
                    f"Error reading from serial port: {e}. Retrying...",
                    RuntimeWarning,
                    stacklevel=2
                )
                continue
    
    def send(self, data : str):
        if not self._check_connected(self._serial):
            raise ConnectionError("Serial connection is not open.")
    
        self._serial.write(data.encode('utf-8'))

    @staticmethod
    def _check_connected(value: Optional[serial.Serial]) -> TypeGuard[serial.Serial]:
        return value is not None and value.is_open