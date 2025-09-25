from typing import Optional

from nn_laser_stabilizer.connection.base_connection import BaseConnection

class PidSerialConnection:
    """
    Обертка над базовым последовательным соединением для отправки PID-команд
    и чтения ответов. Формирует строку команды в формате:
    "kp ki kd u_min u_max\n" с точностью до 4 знаков после запятой
    и разбирает ответы вида "PV CO".
    """

    def __init__(self, connection: BaseConnection):
        self._connection = connection

    def send_pid_command(
        self,
        kp: float,
        ki: float,
        kd: float,
        control_min: float,
        control_max: float,
    ) -> None:
        command = f"{kp:.4f} {ki:.4f} {kd:.4f} {control_min:.4f} {control_max:.4f}"
        self._connection.send_data(command)

    def read_data(self) -> Optional[tuple[float, float]]:
        """
        Читает и парсит ответ измерений из базового соединения.

        Ожидаемый формат: "PV CO" (два числа, разделенные пробелом).
        Возвращает кортеж (process_variable, control_output) или None,
        если данных нет.
        """
        raw = self._connection.read_data()
        if not raw:
            return None
        parts = raw.strip().split()
        if len(parts) != 2:
            raise ValueError(f"Некорректный формат ответа PID: '{raw}'")
        try:
            return float(parts[0]), float(parts[1])
        except Exception as ex:
            raise ValueError(f"Некорректные числовые значения в ответе PID: '{raw}'") from ex


