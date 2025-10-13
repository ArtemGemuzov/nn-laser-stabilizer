from typing import Protocol, Optional


class BaseConnectionToPid(Protocol):
    """
    Протокол для классов, обеспечивающих взаимодействие с PID контроллером.
    
    Определяет интерфейс для отправки PID команд и чтения ответов.
    """
    
    def send_command(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
    ) -> None:
        """Отправляет PID команду контроллеру."""
        ...
    
    def read_response(self) -> tuple[float, float]:
        """
        Блокирующее чтение ответа от контроллера.
        
        Returns:
            tuple[float, float]: Кортеж (process_variable, control_output)
        """
        ...
    
    def exchange(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
    ) -> tuple[float, float]:
        """
        Отправляет PID команду и блокирующе читает ответ.
        
        Returns:
            tuple[float, float]: Кортеж (process_variable, control_output)
        """
        ...

