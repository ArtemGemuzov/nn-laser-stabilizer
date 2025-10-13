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
            - timeout: float - таймаут (по умолчанию 0.1)
            - baudrate: int - скорость передачи (по умолчанию 115200)
            - bytesize: int - размер байта (по умолчанию serial.EIGHTBITS)
            - parity: str - четность (по умолчанию serial.PARITY_NONE)
            - stopbits: int - стоп-биты (по умолчанию serial.STOPBITS_ONE)
    
    Returns:
        BaseConnection: Экземпляр SerialConnection или MockSerialConnection
    """
    serial_config = config.serial
    
    if serial_config.use_mock:
        return MockSerialConnection(
            port=serial_config.port,
            timeout=serial_config.timeout,
            baudrate=serial_config.baudrate,
            bytesize=serial_config.bytesize,
            parity=serial_config.parity,
            stopbits=serial_config.stopbits,
        )
    else:
        return SerialConnection(
            port=serial_config.port,
            timeout=serial_config.timeout,
            baudrate=serial_config.baudrate,
            bytesize=serial_config.bytesize,
            parity=serial_config.parity,
            stopbits=serial_config.stopbits,
        )


__all__ = [
    'BaseConnection',
    'SerialConnection',
    'MockSerialConnection',
    'ConnectionToPid',
    'create_connection',
]

