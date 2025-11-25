from typing import Protocol, Optional

import serial


class BaseConnection(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def send(self, data: str) -> None: ...
    def read(self) -> Optional[str]: ...


class SerialConnection(BaseConnection):
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
        
        self._serial_connection : serial.Serial = None

    def open(self):
        try:
            self._serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=self.bytesize,
                parity=self.parity,
                stopbits=self.stopbits
            )
            if not self._serial_connection.is_open:
                raise ConnectionError("Failed to open serial port.")
        except Exception as ex:
            raise ConnectionError("Error initializing serial connection") from ex

    def close(self):
        if self._serial_connection is not None and self._serial_connection.is_open:
            self._serial_connection.close()

    def _check_connected(self) -> None:
        if not self._serial_connection or not self._serial_connection.is_open:
            raise ConnectionError("Serial connection is not open.")

    def read(self) -> str | None:
        self._check_connected()
        
        while True:
            raw = self._serial_connection.readline().decode("utf-8").strip()
            if raw:
                return raw
    
    def send(self, data : str):
        self._check_connected()
    
        self._serial_connection.write(data.encode('utf-8'))
        

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

    def open(self):
        self.is_connected = True
    
        log_filename = f"mock_{self.port}.log"
        self._log_file = open(log_filename, 'a', encoding='utf-8')
        self._log_file.write(f"Mock serial connection opened")
      

    def close(self):
        self.is_connected = False
        self._log_file.write("Mock serial connection closed.")
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def read(self) -> Optional[str]:
        if not self.is_connected:
            raise ConnectionError("Mock serial connection is not open.")
        
        return None

    def send(self, data: str):
        if not self.is_connected:
            raise ConnectionError("Mock serial connection is not open.")

        self._send_count += 1
        self._log_file.write(f"#{self._send_count} >> {repr(data)}\n")
        self._log_file.flush()