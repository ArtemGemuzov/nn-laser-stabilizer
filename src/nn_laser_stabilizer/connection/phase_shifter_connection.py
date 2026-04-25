import json
import warnings

from nn_laser_stabilizer.hardware.connection import BaseConnection
from nn_laser_stabilizer.connection.phase_shifter_protocol import PhaseShifterProtocol
from nn_laser_stabilizer.utils.logger import Logger


class ConnectionToPhaseShifter:
    MAX_EXCHANGE_ATTEMPTS = 10

    def __init__(self, connection: BaseConnection):
        self._connection = connection

    def open(self) -> None:
        self._connection.open()

    def close(self) -> None:
        self._connection.close()

    def send_command(self, *, control_output: int) -> None:
        command = PhaseShifterProtocol.format_command(control_output)
        self._connection.send(command)

    def read_response(self) -> int:
        raw = self._connection.read()
        return PhaseShifterProtocol.parse_response(raw)

    def exchange(self, *, control_output: int) -> int:
        last_error: Exception | None = None
        for attempt in range(1, self.MAX_EXCHANGE_ATTEMPTS + 1):
            self.send_command(control_output=control_output)
            try:
                return self.read_response()
            except (ValueError, TimeoutError, ConnectionError) as e:
                last_error = e
                warnings.warn(
                    f"Exchange failed (attempt {attempt}/"
                    f"{self.MAX_EXCHANGE_ATTEMPTS}) for control_output="
                    f"{control_output}: {e}. Retrying full exchange...",
                    RuntimeWarning,
                    stacklevel=2,
                )
        assert last_error is not None
        raise ValueError(
            f"Failed to complete exchange after {self.MAX_EXCHANGE_ATTEMPTS} attempts "
            f"for control_output={control_output}"
        ) from last_error


class LoggingConnectionToPhaseShifter(ConnectionToPhaseShifter):
    LOG_SOURCE = "phase_shifter"

    def __init__(
        self,
        connection_to_phase_shifter: ConnectionToPhaseShifter,
        logger: Logger,
    ):
        self._phase_shifter = connection_to_phase_shifter
        self._logger = logger

    def open(self) -> None:
        self._phase_shifter.open()

    def close(self) -> None:
        self._phase_shifter.close()

    def send_command(self, *, control_output: int) -> None:
        self._logger.log(json.dumps({
            "source": self.LOG_SOURCE,
            "event": "send",
            "control_output": control_output,
        }))
        self._phase_shifter.send_command(control_output=control_output)

    def read_response(self) -> int:
        process_variable = self._phase_shifter.read_response()
        self._logger.log(json.dumps({
            "source": self.LOG_SOURCE,
            "event": "read",
            "process_variable": process_variable,
        }))
        return process_variable

    def exchange(self, *, control_output: int) -> int:
        process_variable = self._phase_shifter.exchange(control_output=control_output)
        self._logger.log(json.dumps({
            "source": self.LOG_SOURCE,
            "event": "exchange",
            "control_output": control_output,
            "process_variable": process_variable,
        }))
        return process_variable

