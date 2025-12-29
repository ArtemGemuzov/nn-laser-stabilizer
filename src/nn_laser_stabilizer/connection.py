from typing import Protocol
import warnings
import socket

import serial


class BaseConnection(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def send(self, data: str) -> None: ...
    def read(self) -> str: ...


class COMConnection(BaseConnection):
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
        if self._is_open():
            return
        
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
        if self._is_open():
            self._serial_connection.close()
        self._serial_connection = None

    def read(self) -> str:
        self._check_connected()
        
        while True:
            try:
                raw_data = self._serial_connection.readline()
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
        self._check_connected()
    
        self._serial_connection.write(data.encode('utf-8'))

    def _check_connected(self) -> None:
        if not self._is_open():
            raise ConnectionError("Serial connection is not open.")

    def _is_open(self) -> bool:
        return self._serial_connection is not None and self._serial_connection.is_open
        

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

        log_filename = f"mock_{self.port}.log"
        self._log_file = open(log_filename, 'a', encoding='utf-8')
        self._log_file.write(f"Mock serial connection opened")

        self._read_count = 0
        self._send_count = 0

    def open(self):
        if self.is_connected:
            return
        
        self.is_connected = True
        self._log_file.write(f"Mock serial connection opened")
      
    def close(self):
        if self.is_connected and self._log_file:
            self._log_file.write("Mock serial connection closed.")
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        self.is_connected = False

    def read(self) -> str:
        self._check_connected()

        return "0 0"

    def send(self, data: str):
        self._check_connected()

        self._send_count += 1
        self._log_file.write(f"#{self._send_count} >> {repr(data)}\n")
        self._log_file.flush()

    def _check_connected(self) -> None:
        if not self.is_connected:
            raise ConnectionError("Serial connection is not open.")


class TCPConnection(BaseConnection):
    RECV_BUFFER_SIZE = 4096
    
    def __init__(self,
                 host: str = "localhost",
                 port: int = 8080,
                 timeout: float = 0.1):
        self.host = host
        self.port = port
        self.timeout = timeout
        
        self._socket: socket.socket = None
        self._buffer = b""

    def open(self):
        if self._is_open():
            return
        
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.host, self.port))
        except socket.error as ex:
            raise ConnectionError(f"Error connecting to {self.host}:{self.port}") from ex
        except Exception as ex:
            raise ConnectionError(f"Error initializing TCP connection") from ex

    def close(self):
        if self._is_open():
            self._socket.close()

        self._socket = None
        self._buffer = b""

    def read(self) -> str:
        self._check_connected()
        
        while True:
            try:
                if b"\n" in self._buffer:
                    line, self._buffer = self._buffer.split(b"\n", 1)
                    try:
                        data = line.decode("utf-8")
                        if data:
                            return data
                    except UnicodeDecodeError as e:
                        warnings.warn(
                            f"Failed to decode data from TCP socket: {e}. "
                            f"Raw data: {line!r}. Retrying...",
                            RuntimeWarning,
                            stacklevel=2
                        )
                        continue
                
                raw_data = self._socket.recv(self.RECV_BUFFER_SIZE)
                if not raw_data:
                    raise ConnectionError("TCP connection closed by peer")
                
                self._buffer += raw_data
                
            except socket.timeout:
                continue
            except socket.error as e:
                warnings.warn(
                    f"Error reading from TCP socket: {e}. Retrying...",
                    RuntimeWarning,
                    stacklevel=2
                )
                continue
            except Exception as e:
                warnings.warn(
                    f"Unexpected error reading from TCP socket: {e}. Retrying...",
                    RuntimeWarning,
                    stacklevel=2
                )
                continue
    
    def send(self, data: str):
        self._check_connected()
        
        encoded_data = data.encode('utf-8')
        self._socket.sendall(encoded_data)

    def _check_connected(self) -> None:
        if not self._is_open():
            raise ConnectionError("TCP connection is not open.")

    def _is_open(self) -> bool:
        if self._socket is None:
            return False
        
        try:
            return self._socket.fileno() != -1
        except (socket.error, OSError):
            return False


def parse_tcp_port(port_str: str) -> tuple[str, int]:
    if ":" not in port_str:
        raise ValueError(f"Invalid TCP port format: '{port_str}'. Expected 'host:port' (e.g., 'localhost:8080')")
    
    host, port_str = port_str.rsplit(":", 1)
    if not host:
        raise ValueError(f"Invalid host in TCP port: '{port_str}'. Host cannot be empty")
    
    try:
        port = int(port_str)
    except ValueError:
        raise ValueError(f"Invalid TCP port number in '{port_str}': '{port_str}' is not a valid integer")
    
    return host, port


def create_connection(
    port: str,
    timeout: float = 0.1,
    baudrate: int = 115200,
) -> BaseConnection:
    if port.startswith("COM"):
        return COMConnection(
            port=port,
            timeout=timeout,
            baudrate=baudrate,
        )
    
    host, tcp_port = parse_tcp_port(port)
    return TCPConnection(
        host=host,
        port=tcp_port,
        timeout=timeout,
    )
      