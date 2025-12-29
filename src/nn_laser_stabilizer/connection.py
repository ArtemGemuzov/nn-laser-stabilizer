from typing import Protocol
import warnings
import socket as net

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


class SocketConnection(BaseConnection):
    RECV_BUFFER_SIZE = 4096
    
    def __init__(self, sock: net.socket):
        self._socket = sock
        self._buffer = b""
    
    def open(self) -> None:
        """Сокет уже открыт, ничего не делаем."""
        pass
    
    def close(self) -> None:
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
                            f"Failed to decode data from socket: {e}. "
                            f"Raw data: {line!r}. Retrying...",
                            RuntimeWarning,
                            stacklevel=2
                        )
                        continue
                
                raw_data = self._socket.recv(self.RECV_BUFFER_SIZE)
                if not raw_data:
                    raise ConnectionError("Socket connection closed by peer")
                
                self._buffer += raw_data
                
            except net.timeout:
                continue
            except net.error as e:
                warnings.warn(
                    f"Error reading from socket: {e}. Retrying...",
                    RuntimeWarning,
                    stacklevel=2
                )
                continue
            except Exception as e:
                warnings.warn(
                    f"Unexpected error reading from socket: {e}. Retrying...",
                    RuntimeWarning,
                    stacklevel=2
                )
                continue
    
    def send(self, data: str) -> None:
        self._check_connected()
        
        encoded_data = data.encode('utf-8')
        self._socket.sendall(encoded_data)
    
    def _check_connected(self) -> None:
        if not self._is_open():
            raise ConnectionError("Socket connection is not open.")
    
    def _is_open(self) -> bool:
        if self._socket is None:
            return False
        
        try:
            return self._socket.fileno() != -1
        except (net.error, OSError):
            return False
        

class TCPConnection(BaseConnection):
    def __init__(self,
                 host: str = "localhost",
                 port: int = 8080,
                 timeout: float = 0.1):
        self.host = host
        self.port = port
        self.timeout = timeout
        
        self._socket_connection: SocketConnection | None = None

    def open(self):
        if self._is_open():
            return
        
        try:
            socket = net.socket(socket.AF_INET, socket.SOCK_STREAM)
            socket.settimeout(self.timeout)
            socket.connect((self.host, self.port))

            self._socket_connection = SocketConnection(socket)
        except net.error as ex:
            raise ConnectionError(f"Error connecting to {self.host}:{self.port}") from ex
        except Exception as ex:
            raise ConnectionError(f"Error initializing TCP connection") from ex

    def close(self):
        if self._socket_connection is not None:
            self._socket_connection.close()
        self._socket_connection = None

    def read(self) -> str:
        self._check_connected()
        return self._socket_connection.read()
    
    def send(self, data: str):
        self._check_connected()
        self._socket_connection.send(data)

    def _check_connected(self) -> None:
        if self._socket_connection is None:
            raise ConnectionError("TCP connection is not open.")

    def _is_open(self) -> bool:
        return self._socket_connection is not None


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
    # TODO: возможен порт типа COM:1234
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
      