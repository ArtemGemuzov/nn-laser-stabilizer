import serial
from omegaconf import DictConfig

from nn_laser_stabilizer.connection.base_connection import BaseConnection
from nn_laser_stabilizer.connection.serial_connection import SerialConnection
from nn_laser_stabilizer.connection.mock_serial_connection import MockSerialConnection
from nn_laser_stabilizer.connection.connection_to_pid import ConnectionToPid


def create_connection(config: DictConfig) -> BaseConnection:
    """
    Создает соединение на основе конфигурации.
    
    Args:
        config: Полная конфигурация, содержащая секцию 'serial' с параметрами:
            - use_mock: bool - использовать ли mock соединение
            - port: str - COM порт или имя файла для mock
            - timeout: float - таймаут
            - baudrate: int - скорость передачи
    
    Returns:
        BaseConnection: Экземпляр SerialConnection или MockSerialConnection
    """
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
    'ConnectionToPid',
    'create_connection',
]

