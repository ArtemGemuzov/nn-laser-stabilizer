from nn_laser_stabilizer.hardware.connection import create_connection
from nn_laser_stabilizer.connection.phase_shifter_connection import (
    ConnectionToPhaseShifter,
    LoggingConnectionToPhaseShifter,
)
from nn_laser_stabilizer.logger import Logger, PrefixedLogger


class NeuralControllerPhys:
    LOG_PREFIX = "NEURAL_CONTROLLER_PHYS"

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

        if self._auto_determine_setpoint and self._setpoint_determination_steps <= 1:
            raise ValueError(
                f"setpoint_determination_steps ({self._setpoint_determination_steps}) "
                f"must be greater than 1 when auto_determine_setpoint is True"
            )

        self._base_logger = base_logger
        self._logger = PrefixedLogger(base_logger, NeuralControllerPhys.LOG_PREFIX)

        connection = create_connection(
            port=port,
            timeout=timeout,
            baudrate=baudrate,
        )
        self._pid_connection =  ConnectionToPhaseShifter(connection=connection)

    @property
    def setpoint(self) -> int:
        return self._setpoint

    def _determine_setpoint(self) -> None:
        min_pv = float("inf")
        max_pv = float("-inf")

        for step in range(self._setpoint_determination_steps):
            progress = step / (self._setpoint_determination_steps - 1)
            control_output = int(progress * self._setpoint_determination_max_value)

            process_variable = self._pid_connection.exchange(control_output=control_output)

            min_pv = min(min_pv, process_variable)
            max_pv = max(max_pv, process_variable)

        min_pv_int = int(min_pv)
        max_pv_int = int(max_pv)
        setpoint = int(
            round(min_pv + self._setpoint_determination_factor * (max_pv - min_pv))
        )

        self._setpoint = setpoint
        self._setpoint_determined = True

        self._logger.log(
            f"setpoint determined: setpoint={setpoint} min_pv={min_pv_int} max_pv={max_pv_int}"
        )

        # TODO: Заменить print на ConsoleLogger для унифицированного вывода в консоль
        print(f"Setpoint determined: {self._setpoint} (min_pv={min_pv_int}, max_pv={max_pv_int})")


    def open_and_warmup(self) -> None:
        self._pid_connection.open()

        if self._auto_determine_setpoint and not self._setpoint_determined:
            self._determine_setpoint()

    def neutral_measure(self) -> tuple[int, int]:
        neutral_control_output = (self._control_min + self._control_max) // 2
        process_variable = self._pid_connection.exchange(control_output=neutral_control_output)
        return process_variable, neutral_control_output

    def step(self, control_output: int) -> int:
        return self._pid_connection.exchange(control_output=control_output)

    def close(self) -> None:
        self._pid_connection.close()

