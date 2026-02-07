from typing import Callable, Protocol, Tuple

from nn_laser_stabilizer.rl.envs.setpoint import determine_setpoint
from nn_laser_stabilizer.hardware.connection import create_connection
from nn_laser_stabilizer.connection.phase_shifter_connection import (
    ConnectionToPhaseShifter,
    LoggingConnectionToPhaseShifter,
)
from nn_laser_stabilizer.logger import Logger, PrefixedLogger


class PlantBackend(Protocol):
    @property
    def setpoint(self) -> int:
        ...

    def reset(self) -> Tuple[int, int, int]:
        """Сбросить установку к началу эпизода; возвращает (process_variable, setpoint, control_output)."""
        ...

    def exchange(self, control_output: int) -> int:
        """Отправить управление, получить текущую process_variable."""
        ...

    def close(self) -> None:
        ...


class ExperimentalPlantBackend:
    LOG_PREFIX = "EXPERIMENTAL_PLANT_BACKEND"

    def __init__(
        self,
        *,
        # Параметры для соединения
        port: str,
        timeout: float,
        baudrate: int,
        # Параметры для работы с установкой
        setpoint: int,
        # Параметры автоматического определения setpoint
        auto_determine_setpoint: bool,
        setpoint_determination_steps: int,
        setpoint_determination_max_value: int,
        setpoint_determination_factor: float,
        # Параметры диапазона управления
        control_min: int,
        control_max: int,
        # Сброс: фиксированное значение и число шагов в начале эпизода
        reset_value: int,
        reset_steps: int,
        # Логирование соединения
        log_connection: bool,
        # Логгер верхнего уровня
        base_logger: Logger,
    ):
        self._setpoint: int = setpoint
        self._auto_determine_setpoint = auto_determine_setpoint
        self._setpoint_determination_steps = setpoint_determination_steps
        self._setpoint_determination_max_value = setpoint_determination_max_value
        self._setpoint_determination_factor = setpoint_determination_factor
        self._setpoint_determined = False

        self._control_min = control_min
        self._control_max = control_max

        self._reset_value = int(reset_value)
        self._reset_steps = int(reset_steps)

        if self._auto_determine_setpoint and self._setpoint_determination_steps <= 1:
            raise ValueError(
                f"setpoint_determination_steps ({self._setpoint_determination_steps}) "
                f"must be greater than 1 when auto_determine_setpoint is True"
            )

        self._base_logger = base_logger
        self._logger = PrefixedLogger(base_logger, ExperimentalPlantBackend.LOG_PREFIX)

        connection = create_connection(
            port=port,
            timeout=timeout,
            baudrate=baudrate,
        )
        pid_connection = ConnectionToPhaseShifter(connection=connection)
        if log_connection:
            pid_connection = LoggingConnectionToPhaseShifter(
                connection_to_phase_shifter=pid_connection,
                logger=base_logger,
            )
        self._pid_connection = pid_connection

    @property
    def setpoint(self) -> int:
        return self._setpoint

    def _determine_setpoint(self) -> None:
        setpoint, min_pv_int, max_pv_int = determine_setpoint(
            send_control_and_get_pv=lambda c: self._pid_connection.exchange(control_output=c),
            steps=self._setpoint_determination_steps,
            max_value=self._setpoint_determination_max_value,
            factor=self._setpoint_determination_factor,
        )
        self._setpoint = setpoint
        self._setpoint_determined = True

        self._logger.log(
            f"setpoint determined: setpoint={setpoint} min_pv={min_pv_int} max_pv={max_pv_int}"
        )

        # TODO: Заменить print на ConsoleLogger для унифицированного вывода в консоль
        print(f"Setpoint determined: {self._setpoint} (min_pv={min_pv_int}, max_pv={max_pv_int})")

    def reset(self) -> tuple[int, int, int]:
        self._pid_connection.open()

        if self._auto_determine_setpoint and not self._setpoint_determined:
            self._determine_setpoint()

        process_variable = 0
        for _ in range(self._reset_steps):
            process_variable = self._pid_connection.exchange(
                control_output=self._reset_value
            )

        return process_variable, self._setpoint, self._reset_value

    def exchange(self, control_output: int) -> int:
        return self._pid_connection.exchange(control_output=control_output)

    def close(self) -> None:
        self._pid_connection.close()


class MockPlantBackend:
    def __init__(
        self,
        *,
        reset_fn: Callable[[], tuple[int, int, int]],
        exchange_fn: Callable[[int], int],
        setpoint: int,
    ):
        self._reset_fn = reset_fn
        self._exchange_fn = exchange_fn
        self._setpoint = setpoint

    @property
    def setpoint(self) -> int:
        return self._setpoint

    def reset(self) -> tuple[int, int, int]:
        return self._reset_fn()

    def exchange(self, control_output: int) -> int:
        return self._exchange_fn(control_output)

    def close(self) -> None:
        pass
