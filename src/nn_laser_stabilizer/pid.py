from typing import Protocol, Optional
import math
import random

from nn_laser_stabilizer.connection import BaseConnection
from nn_laser_stabilizer.logger import AsyncFileLogger


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

    def _format_command(self, *, kp: float, ki: float, kd: float, control_min: int, control_max: int) -> str:
        return f"{kp:.4f} {ki:.4f} {kd:.4f} {control_min:.1f} {control_max:.1f}\n" 

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
        self._connection.send(command)

    def _parse_response(self, raw: str) -> tuple[float, float]:
        parts = raw.strip().split()
        if len(parts) != 2:
            raise ValueError(f"Invalid PID response format: '{raw}'")
        return float(parts[0]), float(parts[1])
       
    def read_response(self) -> tuple[float, float]:
        while True:
            raw = self._connection.read()
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
    
    def open(self) -> None:
        self._logger.log("OPEN: Opening connection to PID controller")
        self._pid.open()
        self._logger.log("OPEN: Connection opened successfully")
    
    def close(self) -> None:
        self._logger.log("CLOSE: Closing connection to PID controller")
        self._pid.close()
        self._logger.log("CLOSE: Connection closed successfully")
    
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
            f"SEND: kp={kp:.4f} ki={ki:.4f} kd={kd:.4f} "
            f"control_min={control_min} control_max={control_max}"
        )
        self._pid.send_command(kp=kp, ki=ki, kd=kd, control_min=control_min, control_max=control_max)
    
    def read_response(self) -> tuple[float, float]:
        process_variable, control_output = self._pid.read_response()
        self._logger.log(
            f"READ: process_variable={process_variable:.4f} control_output={control_output:.4f}"
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


class TestConnectionToPid(ConnectionToPid):
    MAX_DISTANCE: float = 20.0
    NOISE_STD: float = 30.0
    
    def __init__(
        self,
        connection: BaseConnection,
        kp_min: float,
        kp_max: float,
        ki_min: float,
        ki_max: float,
        kd_min: float,
        kd_max: float,
        setpoint: float,
    ):
        super().__init__(connection)
        self._optimal_kp = kp_min + (kp_max - kp_min) * 0.75
        self._optimal_ki = ki_min + (ki_max - ki_min) * 0.75
        self._optimal_kd = kd_min + (kd_max - kd_min) * 0.75
        self._setpoint = setpoint
        
        self._last_kp: Optional[float] = None
        self._last_ki: Optional[float] = None
        self._last_kd: Optional[float] = None
        self._last_control_min: Optional[float] = None
        self._last_control_max: Optional[float] = None
    
    def send_command(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
    ) -> None:
        self._last_kp = kp
        self._last_ki = ki
        self._last_kd = kd
        self._last_control_min = float(control_min)
        self._last_control_max = float(control_max)
        
        super().send_command(
            kp=kp,
            ki=ki,
            kd=kd,
            control_min=control_min,
            control_max=control_max,
        )
    
    def read_response(self) -> tuple[float, float]:
        if self._last_kp is None or self._last_ki is None or self._last_kd is None:
            raise RuntimeError("No command was sent before reading response")
        
        distance = math.sqrt(
            (self._last_kp - self._optimal_kp) ** 2 + 
            (self._last_ki - self._optimal_ki) ** 2 +
            (self._last_kd - self._optimal_kd) ** 2
        )
        closeness = max(0.0, 1.0 - min(distance, self.MAX_DISTANCE) / self.MAX_DISTANCE)
        
        random_component = random.randint(0, 2000)
        noise = random.gauss(0, self.NOISE_STD)
        process_variable = (
            closeness * self._setpoint + 
            (1.0 - closeness) * random_component + 
            noise
        )
        process_variable = int(max(0, min(2000, round(process_variable))))
        
        control_min = self._last_control_min or 0.0
        control_max = self._last_control_max or 4095.0
        control_output = random.randint(int(control_min), int(control_max))
        
        return float(process_variable), float(control_output)