from typing import Protocol


from nn_laser_stabilizer.hardware.com_port import COMPort
from nn_laser_stabilizer.hardware.socket import Socket, parse_socket_port


class BaseConnection(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def send(self, data: str) -> None: ...
    def read(self) -> str: ...


def create_connection(
    port: str,
    timeout: float = 0.1,
    baudrate: int = 115200,
) -> BaseConnection:
    if port.startswith("COM") and ":" in port:
        raise ValueError(
            f"Ambiguous port format: '{port}'. "
            f"Cannot determine if this is a COM port or TCP port. "
            f"Use 'COM1', 'COM3' for COM ports or 'host:port' (e.g., 'localhost:8080') for TCP ports"
        )
    
    if ":" in port:
        host, tcp_port = parse_socket_port(port)
        return Socket(
            host=host,
            port=tcp_port,
            timeout=timeout,
        )
    
    if port.startswith("COM"):
        return COMPort(
            port=port,
            timeout=timeout,
            baudrate=baudrate,
        )
    
    raise ValueError(
        f"Invalid port format: '{port}'. "
        f"Expected COM port (e.g., 'COM1', 'COM3') or TCP port (e.g., 'localhost:8080')"
    )
      