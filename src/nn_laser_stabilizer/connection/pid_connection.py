from typing import Protocol

from nn_laser_stabilizer.hardware.connection import BaseConnection
from nn_laser_stabilizer.connection.pid_protocol import PidProtocol
from nn_laser_stabilizer.logger import Logger, PrefixedLogger


class ConnectionToPidProtocol(Protocol):  
    def open(self) -> None: ...
    
    def close(self) -> None: ...
    
    def send_command(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
        setpoint: int,
    ) -> None: ...
    
    def read_response(self) -> tuple[float, float]: ...
    
    def exchange(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
        setpoint: int,
    ) -> tuple[float, float]: ...


class ConnectionToPid:
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
        setpoint: int,
    ) -> None:
        command = PidProtocol.format_command(
            kp=kp, ki=ki, kd=kd, 
            control_min=control_min, control_max=control_max,
            setpoint=setpoint
        )
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
        setpoint: int,
    ) -> tuple[float, float]:
        self.send_command(
            kp=kp,
            ki=ki,
            kd=kd,
            control_min=control_min,
            control_max=control_max,
            setpoint=setpoint,
        )
        return self.read_response()
    

class LoggingConnectionToPid(ConnectionToPid):
    LOG_PREFIX = "PID"
    
    def __init__(
        self,
        connection_to_pid: ConnectionToPid,
        logger: Logger,
    ):
        self._pid = connection_to_pid
        self._logger = PrefixedLogger(logger, LoggingConnectionToPid.LOG_PREFIX)
    
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
        setpoint: int,
    ) -> None:
        self._logger.log(
            f"send: kp={kp} ki={ki} kd={kd} "
            f"control_min={control_min} control_max={control_max} setpoint={setpoint}"
        )
        self._pid.send_command(
            kp=kp, ki=ki, kd=kd, 
            control_min=control_min, control_max=control_max,
            setpoint=setpoint
        )
    
    def read_response(self) -> tuple[float, float]:
        process_variable, control_output = self._pid.read_response()
        self._logger.log(
            f"read: process_variable={process_variable} control_output={control_output}"
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
        setpoint: int,
    ) -> tuple[float, float]:
        self.send_command(
            kp=kp, ki=ki, kd=kd, 
            control_min=control_min, control_max=control_max,
            setpoint=setpoint
        )
        return self.read_response()

