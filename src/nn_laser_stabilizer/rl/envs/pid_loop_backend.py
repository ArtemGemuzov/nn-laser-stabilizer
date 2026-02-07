from typing import Protocol, Tuple

import numpy as np

from nn_laser_stabilizer.rl.envs.setpoint import determine_setpoint
from nn_laser_stabilizer.hardware.connection import create_connection
from nn_laser_stabilizer.connection.pid_connection import (
    ConnectionToPid,
    LoggingConnectionToPid,
    ConnectionToPidProtocol
)
from nn_laser_stabilizer.logger import Logger, PrefixedLogger


class PidLoopBackend(Protocol):
    @property
    def setpoint(self) -> int:
        ...

    def start(self, kp: float, ki: float, kd: float) -> Tuple[np.ndarray, np.ndarray, int, bool]:
        """Начать эпизод с заданными коэффициентами; возвращает (process_variables, control_outputs, setpoint, should_reset)."""
        ...

    def run_block(self, kp: float, ki: float, kd: float) -> Tuple[np.ndarray, np.ndarray, int, bool]:
        """Прогнать один блок с заданными коэффициентами; возвращает (process_variables, control_outputs, setpoint, should_reset)."""
        ...

    def close(self) -> None:
        ...


def determine_setpoint_for_pid_loop(
    pid_connection: ConnectionToPidProtocol,
    steps: int,
    max_value: int,
    factor: float,
) -> tuple[int, int, int]:
    def send_control_and_get_pv(control_value: int) -> int:
        pv, _ = pid_connection.exchange(
            kp=0.0,
            ki=0.0,
            kd=0.0,
            control_min=control_value,
            control_max=control_value,
            setpoint=0,
        )
        return int(pv)

    return determine_setpoint(
        send_control_and_get_pv=send_control_and_get_pv,
        steps=steps,
        max_value=max_value,
        factor=factor,
    )


class ExperimentalPidLoopBackend:
    LOG_PREFIX = "EXPERIMENTAL_PID_LOOP_BACKEND"

    def __init__(
        self,
        *,
        # Логгер верхнего уровня
        logger: Logger,
        # Параметры для соединения
        port: str,
        timeout: float,
        baudrate: int,
        # Параметры для логирования соединения
        log_connection: bool,
        # Параметры для работы с установкой
        setpoint: int,
        warmup_steps: int,
        block_size: int,
        burn_in_steps: int,
        control_output_min_threshold: float,
        control_output_max_threshold: float,
        force_min_value: int,
        force_max_value: int,
        default_min: int,
        default_max: int,
        auto_determine_setpoint: bool,
        setpoint_determination_steps: int,
        setpoint_determination_max_value: int,
        setpoint_determination_factor: float,
    ):
        connection = create_connection(
            port=port,
            timeout=timeout,
            baudrate=baudrate,
        )
        pid_connection: ConnectionToPid | LoggingConnectionToPid = ConnectionToPid(
            connection=connection
        )

        if log_connection:
            pid_connection = LoggingConnectionToPid(
                connection_to_pid=pid_connection,
                logger=logger,
            )

        self.pid_connection = pid_connection

        self._logger = PrefixedLogger(logger, ExperimentalPidLoopBackend.LOG_PREFIX)

        self._warmup_steps = warmup_steps
        self._block_size = block_size

        if burn_in_steps >= block_size:
            raise ValueError(
                f"burn_in_steps ({burn_in_steps}) must be less than "
                f"block_size={block_size}, otherwise no data will remain after burn_in"
            )

        self._burn_in_steps = burn_in_steps

        self._control_output_min_threshold = control_output_min_threshold
        self._control_output_max_threshold = control_output_max_threshold

        self._force_min_value = force_min_value
        self._force_max_value = force_max_value
        self._default_min = default_min
        self._default_max = default_max

        self._process_variables = np.zeros(block_size, dtype=np.float32)
        self._control_outputs = np.zeros(block_size, dtype=np.float32)
        self._current_index = 0

        self._setpoint = setpoint
        self._auto_determine_setpoint = auto_determine_setpoint
        self._setpoint_determination_steps = setpoint_determination_steps
        self._setpoint_determination_max_value = setpoint_determination_max_value
        self._setpoint_determination_factor = setpoint_determination_factor
        self._setpoint_determined = False

        if auto_determine_setpoint and setpoint_determination_steps <= 1:
            raise ValueError(
                f"setpoint_determination_steps ({setpoint_determination_steps}) must be greater than 1 "
                f"when auto_determine_setpoint is True"
            )

    @property
    def setpoint(self) -> int:
        return self._setpoint

    def _should_reset(self, control_outputs: np.ndarray) -> bool:
        mean_control_output = np.mean(control_outputs)
        return bool(mean_control_output < self._control_output_min_threshold or
                mean_control_output > self._control_output_max_threshold)

    def _determine_setpoint(self) -> None:
        setpoint, min_pv_int, max_pv_int = determine_setpoint_for_pid_loop(
            pid_connection=self.pid_connection,
            steps=self._setpoint_determination_steps,
            max_value=self._setpoint_determination_max_value,
            factor=self._setpoint_determination_factor,
        )

        self._setpoint = setpoint
        self._setpoint_determined = True

        self._logger.log(
            f"setpoint determined: setpoint={self._setpoint} "
            f"min_pv={min_pv_int} max_pv={max_pv_int}"
        )

        # TODO: Заменить print на ConsoleLogger для унифицированного вывода в консоль
        print(f"Setpoint determined: {self._setpoint} (min_pv={min_pv_int}, max_pv={max_pv_int})")

    def run_block(self, kp: float, ki: float, kd: float) -> Tuple[np.ndarray, np.ndarray, int, bool]:
        self._reset_buffer()

        for _ in range(self._block_size):
            process_variable, control_output = self.pid_connection.exchange(
                kp=kp,
                ki=ki,
                kd=kd,
                control_min=self._default_min,
                control_max=self._default_max,
                setpoint=self._setpoint,
            )

            self._process_variables[self._current_index] = process_variable
            self._control_outputs[self._current_index] = control_output
            self._current_index += 1

        process_variables, control_outputs = self._get_buffer()
        should_reset = self._should_reset(control_outputs)

        return process_variables, control_outputs, self._setpoint, should_reset

    def _get_buffer(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self._process_variables[self._burn_in_steps:self._current_index],
            self._control_outputs[self._burn_in_steps:self._current_index]
        )

    def _reset_buffer(self) -> None:
        self._current_index = 0

    def start(self, kp: float, ki: float, kd: float) -> Tuple[np.ndarray, np.ndarray, int, bool]:
        self.pid_connection.open()

        if self._auto_determine_setpoint and not self._setpoint_determined:
            self._determine_setpoint()

        for _ in range(self._warmup_steps):
            self.pid_connection.exchange(
                kp=kp,
                ki=ki,
                kd=kd,
                control_min=self._force_min_value,
                control_max=self._force_max_value,
                setpoint=self._setpoint,
            )

        return self.run_block(kp, ki, kd)

    def close(self) -> None:
        self.pid_connection.close()
