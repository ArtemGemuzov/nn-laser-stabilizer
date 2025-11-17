from typing import Protocol, Optional


class BaseConnectionToPid(Protocol):
    """
    Протокол для классов, обеспечивающих взаимодействие с PID контроллером.
    
    Определяет интерфейс для отправки PID команд и чтения ответов.
    """
    
    def open_connection(self) -> None:
        """Открывает соединение с контроллером."""
        ...
    
    def close_connection(self) -> None:
        """Закрывает соединение с контроллером."""
        ...
    
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

