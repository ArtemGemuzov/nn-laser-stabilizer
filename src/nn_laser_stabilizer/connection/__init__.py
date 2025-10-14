import os
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


def create_connection_to_pid(config: DictConfig, output_dir: str) -> BaseConnectionToPid:
    """
    Создает соединение с PID контроллером на основе конфигурации.
    
    Соединение НЕ открывается автоматически - это делается в ExperimentalSetupController.reset()
    
    Args:
        config: Конфигурация, содержащая:
            - serial.use_mock: использовать ли mock соединение
            - serial.port: COM порт
            - serial.baudrate: скорость передачи (опционально)
            - serial.timeout: таймаут (опционально)
            - serial.log_connection: логировать ли команды и ответы
        output_dir: Директория для логов соединения
    
    Returns:
        BaseConnectionToPid: Настроенное (но не открытое) соединение с PID контроллером
    """
    from nn_laser_stabilizer.logging.async_file_logger import AsyncFileLogger
    
    serial_connection = create_connection(config)
    # НЕ вызываем open_connection() здесь - это делает контроллер
    
    pid_connection = ConnectionToPid(serial_connection)
    
    if config.serial.log_connection:
        connection_log_dir = os.path.join(output_dir, "connection_logs")
        connection_logger = AsyncFileLogger(log_dir=connection_log_dir, filename="connection.log")
        pid_connection = LoggingConnectionToPid(pid_connection, connection_logger)
    
    return pid_connection


__all__ = [
    'BaseConnection',
    'SerialConnection',
    'MockSerialConnection',
    'BaseConnectionToPid',
    'ConnectionToPid',
    'LoggingConnectionToPid',
    'create_connection',
    'create_connection_to_pid',
]

