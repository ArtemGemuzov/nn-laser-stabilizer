from typing import Optional, TypeGuard
import warnings
import socket as net


def parse_socket_port(port_str: str) -> tuple[str, int]:
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


class SocketAdapter:
    RECV_BUFFER_SIZE = 4096
    
    def __init__(self, sock: net.socket):
        self._socket = sock
        self._buffer = b""
    
    def open(self) -> None:
        """Сокет уже открыт, ничего не делаем."""
        ...
    
    def close(self) -> None:
        if self._check_connected(self._socket):
            self._socket.close()
        self._socket = None
        self._buffer = b""
    
    def read(self) -> str:
        if not self._check_connected(self._socket):
            raise ConnectionError("Socket connection is not open.")
        
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
        if not self._check_connected(self._socket):
            raise ConnectionError("Socket connection is not open.")
        
        encoded_data = data.encode('utf-8')
        self._socket.sendall(encoded_data)
    
    @staticmethod
    def _check_connected(value: Optional[net.socket]) -> TypeGuard[net.socket]:
        if value is None:
            return False
        try:
            return value.fileno() != -1
        except (net.error, OSError):
            return False
        

class Socket:
    def __init__(self,
                 host: str = "localhost",
                 port: int = 8080,
                 timeout: float = 0.1):
        self.host = host
        self.port = port
        self.timeout = timeout
        
        self._socket: SocketAdapter | None = None

    def open(self):
        if self._check_connected(self._socket):
            return
        
        try:
            socket = net.socket(net.AF_INET, net.SOCK_STREAM)
            socket.settimeout(self.timeout)
            socket.connect((self.host, self.port))

            self._socket = SocketAdapter(socket)
        except net.error as ex:
            raise ConnectionError(f"Error connecting to {self.host}:{self.port}") from ex
        except Exception as ex:
            raise ConnectionError(f"Error initializing TCP connection") from ex

    def close(self):
        if self._check_connected(self._socket):
            self._socket.close()
        self._socket = None

    def read(self) -> str:
        if not self._check_connected(self._socket):
            raise ConnectionError("TCP connection is not open.")
        return self._socket.read()
    
    def send(self, data: str):
        if not self._check_connected(self._socket):
            raise ConnectionError("TCP connection is not open.")
        self._socket.send(data)

    @staticmethod
    def _check_connected(value: Optional[SocketAdapter]) -> TypeGuard[SocketAdapter]:
        return value is not None