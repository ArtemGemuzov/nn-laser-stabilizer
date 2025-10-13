from typing import Optional

from nn_laser_stabilizer.connection.base_connection import BaseConnection

class ConnectionToPid:
    """
    Обертка над базовым последовательным соединением для отправки PID-команд
    и чтения ответов. Формирует строку команды в формате:
    "kp ki kd u_min u_max" с точностью до 4 знаков после запятой
    и разбирает ответы вида "process_variable control_output".
    """

    def __init__(self, connection: BaseConnection):
        self._connection = connection

    def _format_command(self, kp: float, ki: float, kd: float, control_min: int, control_max: int) -> str:
        """Форматирует команду PID в строку."""
        return f"{kp:.4f} {ki:.4f} {kd:.4f} {control_min} {control_max}\n"

    def send_pid_command(
        self,
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
    ) -> None:
        command = self._format_command(kp, ki, kd, control_min, control_max)
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

    def read_data(self) -> Optional[tuple[float, float]]:
        """
        Читает и парсит ответ измерений из базового соединения.

        Ожидаемый формат: "process_variable control_output" (два числа, разделенные пробелом).
        Возвращает кортеж (process_variable, control_output) или None,
        если данных нет.
        """
        raw = self._connection.read_data()
        if not raw:
            return None
        return self._parse_response(raw)

    def read_data_and_wait(self) -> tuple[float, float]:
        """
        Блокирующее чтение до получения корректного ответа PID.

        Возвращает кортеж (process_variable, control_output).
        """
        while True:
            data = self.read_data()
            if data is not None:
                return data

    def send_and_read(
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
        self.send_pid_command(
            kp=kp,
            ki=ki,
            kd=kd,
            control_min=control_min,
            control_max=control_max,
        )
        return self.read_data_and_wait()


