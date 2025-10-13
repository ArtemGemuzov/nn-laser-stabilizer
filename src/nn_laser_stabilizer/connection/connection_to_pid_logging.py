from typing import Optional

from nn_laser_stabilizer.connection import BaseConnectionToPid, ConnectionToPid
from nn_laser_stabilizer.logging.async_file_logger import AsyncFileLogger


class LoggingConnectionToPid(BaseConnectionToPid):
    """
    Обертка над ConnectionToPid с логированием всех команд и ответов.
    
    Логирует отправленные PID команды (kp, ki, kd, control_min, control_max)
    и полученные ответы (process_variable, control_output) в структурированном формате.
    """
    
    def __init__(self, connection_to_pid: ConnectionToPid, logger: AsyncFileLogger):
        """
        Args:
            connection_to_pid: Экземпляр ConnectionToPid для оборачивания
            logger: Логгер для записи команд и ответов
        """
        self._pid = connection_to_pid
        self._logger = logger
        self._connection = connection_to_pid._connection
    
    def _log_send_command(
        self,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
    ) -> None:
        """Логирует отправку PID команды."""
        self._logger.log(
            f"SEND: kp={kp:.4f} ki={ki:.4f} kd={kd:.4f} "
            f"control_min={control_min} control_max={control_max}"
        )
    
    def _log_read_response(self, process_variable: float, control_output: float) -> None:
        """Логирует прочитанный ответ от контроллера."""
        self._logger.log(
            f"READ: process_variable={process_variable:.4f} control_output={control_output:.4f}"
        )
    
    def send_command(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
    ) -> None:
        """Отправляет PID команду с логированием."""
        self._log_send_command(kp, ki, kd, control_min, control_max)
        self._pid.send_command(kp=kp, ki=ki, kd=kd, control_min=control_min, control_max=control_max)
    
    def read_response(self) -> tuple[float, float]:
        """Блокирующее чтение с логированием."""
        process_variable, control_output = self._pid.read_response()
        self._log_read_response(process_variable, control_output)
        return process_variable, control_output
    
    def exchange(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
    ) -> tuple[float, float]:
        """Отправляет команду и читает ответ с логированием."""
        self._log_send_command(kp, ki, kd, control_min, control_max)
        process_variable, control_output = self._pid.exchange(
            kp=kp,
            ki=ki,
            kd=kd,
            control_min=control_min,
            control_max=control_max,
        )
        self._log_read_response(process_variable, control_output)
        return process_variable, control_output

