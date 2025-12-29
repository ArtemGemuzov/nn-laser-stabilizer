from typing import Protocol

from nn_laser_stabilizer.connection import BaseConnection
from nn_laser_stabilizer.logger import AsyncFileLogger
from nn_laser_stabilizer.pid_protocol import PidProtocol


class BaseConnectionToPid(Protocol):  
    def open(self) -> None: pass
    
    def close(self) -> None: pass
    
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
    
    def open(self) -> None:
        self._connection.open()
    
    def close(self) -> None:
        self._connection.close()

    def send_command(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
    ) -> None:
        command = PidProtocol.format_command(kp=kp, ki=ki, kd=kd, control_min=control_min, control_max=control_max)
        self._connection.send(command)

    def read_response(self) -> tuple[float, float]:
        raw = self._connection.read()
        return PidProtocol.parse_response(raw)

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
    
    def open(self) -> None:
        self._pid.open()
    
    def close(self) -> None:
        self._pid.close()
    
    def send_command(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
    ) -> None:
        self._logger.log(
            f"SEND: kp={kp} ki={ki} kd={kd} "
            f"control_min={control_min} control_max={control_max}"
        )
        self._pid.send_command(kp=kp, ki=ki, kd=kd, control_min=control_min, control_max=control_max)
    
    def read_response(self) -> tuple[float, float]:
        process_variable, control_output = self._pid.read_response()
        self._logger.log(
            f"READ: process_variable={process_variable} control_output={control_output}"
        )
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
        self.send_command(kp=kp, ki=ki, kd=kd, control_min=control_min, control_max=control_max)
        return self.read_response()

