from typing import Protocol

from nn_laser_stabilizer.connection import BaseConnection
from nn_laser_stabilizer.logger import AsyncFileLogger


class BaseConnectionToPid(Protocol):  
    def open_connection(self) -> None: pass
    
    def close_connection(self) -> None: pass
    
    def send_command(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
    ) -> None: pass
    
    def read_response(self) -> tuple[float, float]: pass
    
    def exchange(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
    ) -> tuple[float, float]: pass


class ConnectionToPid(BaseConnectionToPid):
    def __init__(self, connection: BaseConnection):
        self._connection = connection
    
    def open_connection(self) -> None:
        self._connection.open_connection()
    
    def close_connection(self) -> None:
        self._connection.close_connection()

    def _format_command(self, *, kp: float, ki: float, kd: float, control_min: int, control_max: int) -> str:
        return f"{kp:.4f} {ki:.4f} {kd:.4f} {control_min:.4f} {control_max:.4f}\n" 

    def send_command(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
    ) -> None:
        command = self._format_command(kp=kp, ki=ki, kd=kd, control_min=control_min, control_max=control_max)
        self._connection.send_data(command)

    def _parse_response(self, raw: str) -> tuple[float, float]:
        parts = raw.strip().split()
        if len(parts) != 2:
            raise ValueError(f"Invalid PID response format: '{raw}'")
        try:
            return float(parts[0]), float(parts[1])
        except Exception as ex:
            raise ValueError(f"Invalid numeric values in PID response: '{raw}'") from ex

    def read_response(self) -> tuple[float, float]:
        while True:
            raw = self._connection.read_data()
            if raw is not None:
                return self._parse_response(raw)

    def exchange(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
    ) -> tuple[float, float]:
        self.send_command(
            kp=kp,
            ki=ki,
            kd=kd,
            control_min=control_min,
            control_max=control_max,
        )
        return self.read_response()
    

class LoggingConnectionToPid(BaseConnectionToPid):  
    def __init__(self, connection_to_pid: ConnectionToPid, logger: AsyncFileLogger):
        self._pid = connection_to_pid
        self._logger = logger
    
    def open_connection(self) -> None:
        self._logger.log("OPEN: Opening connection to PID controller")
        self._pid.open_connection()
        self._logger.log("OPEN: Connection opened successfully")
    
    def close_connection(self) -> None:
        self._logger.log("CLOSE: Closing connection to PID controller")
        self._pid.close_connection()
        self._logger.log("CLOSE: Connection closed successfully")
    
    def _log_send_command(
        self,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
    ) -> None:
        self._logger.log(
            f"SEND: kp={kp:.4f} ki={ki:.4f} kd={kd:.4f} "
            f"control_min={control_min} control_max={control_max}"
        )
    
    def _log_read_response(self, process_variable: float, control_output: float) -> None:
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
        self._log_send_command(kp, ki, kd, control_min, control_max)
        self._pid.send_command(kp=kp, ki=ki, kd=kd, control_min=control_min, control_max=control_max)
    
    def read_response(self) -> tuple[float, float]:
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