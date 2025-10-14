from typing import Optional


from nn_laser_stabilizer.connection import BaseConnection, BaseConnectionToPid


class ConnectionToPid(BaseConnectionToPid):
    """
    Обертка над базовым последовательным соединением для отправки PID-команд
    и чтения ответов. Формирует строку команды в формате:
    "kp ki kd u_min u_max" с точностью до 4 знаков после запятой
    и разбирает ответы вида "process_variable control_output".
    """

    def __init__(self, connection: BaseConnection):
        self._connection = connection
    
    def open_connection(self) -> None:
        """Открывает соединение с контроллером."""
        self._connection.open_connection()
    
    def close_connection(self) -> None:
        """Закрывает соединение с контроллером."""
        self._connection.close_connection()

    def _format_command(self, *, kp: float, ki: float, kd: float, control_min: int, control_max: int) -> str:
        """Форматирует команду PID в строку.
        
        Для обратной совместимости границы управляющего сигнала отправляются как float.
        """
        return f"{kp:.4f} {ki:.4f} {kd:.4f} {control_min:4f} {control_max:4f}\n" 

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
        """Парсит ответ PID из строки."""
        parts = raw.strip().split()
        if len(parts) != 2:
            raise ValueError(f"Некорректный формат ответа PID: '{raw}'")
        try:
            return float(parts[0]), float(parts[1])
        except Exception as ex:
            raise ValueError(f"Некорректные числовые значения в ответе PID: '{raw}'") from ex

    def read_response(self) -> tuple[float, float]:
        """
        Блокирующее чтение до получения корректного ответа PID.

        Возвращает кортеж (process_variable, control_output).
        """
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
        """
        Отправляет коэффициенты, затем блокирующе читает ответ.

        Возвращает (PV, CO).
        """
        self.send_command(
            kp=kp,
            ki=ki,
            kd=kd,
            control_min=control_min,
            control_max=control_max,
        )
        return self.read_response()


