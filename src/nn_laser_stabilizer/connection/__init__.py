import serial
from omegaconf import DictConfig

from nn_laser_stabilizer.connection.base_connection import BaseConnection
from nn_laser_stabilizer.connection.serial_connection import SerialConnection
from nn_laser_stabilizer.connection.mock_serial_connection import MockSerialConnection
from nn_laser_stabilizer.connection.base_connection_to_pid import BaseConnectionToPid
from nn_laser_stabilizer.connection.connection_to_pid import ConnectionToPid
from nn_laser_stabilizer.connection.connection_to_pid_logging import LoggingConnectionToPid


def create_connection(config: DictConfig) -> BaseConnection:
    serial_config = config.serial
    
    if serial_config.use_mock:
        return MockSerialConnection(
            port=serial_config.port,
            timeout=serial_config.timeout,
            baudrate=serial_config.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
    else:
        return SerialConnection(
            port=serial_config.port,
            timeout=serial_config.timeout,
            baudrate=serial_config.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )


__all__ = [
    'BaseConnection',
    'SerialConnection',
    'MockSerialConnection',
    'BaseConnectionToPid',
    'ConnectionToPid',
    'LoggingConnectionToPid',
    'create_connection',
]

