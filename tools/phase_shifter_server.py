import argparse
import random

from nn_laser_stabilizer.connection.phase_shifter_protocol import PhaseShifterProtocol
from nn_laser_stabilizer.hardware.socket import parse_socket_port, SocketAdapter
from nn_laser_stabilizer.hardware.server import Server, run_server
from nn_laser_stabilizer.experiment.decorator import experiment
from nn_laser_stabilizer.experiment.context import ExperimentContext


class PhaseShifterSimulator:
    """Симулятор фазовращателя для тестирования окружения."""

    def __init__(
        self,
        control_min: int,
        control_max: int,
        setpoint: int,
        process_variable_max: int,
        noise_std: float = 10.0,
    ):
        self._control_min = control_min
        self._control_max = control_max
        self._setpoint = setpoint
        self._process_variable_max = process_variable_max
        self._noise_std = noise_std

        self._current_pv = setpoint
        self._optimal_control = (control_min + control_max) // 2

    def step(self, control: int) -> int:
        control_norm = (control - self._control_min) / (self._control_max - self._control_min)
        optimal_norm = (self._optimal_control - self._control_min) / (self._control_max - self._control_min)
        distance = abs(control_norm - optimal_norm)
        closeness = 1.0 - min(distance, 1.0)
        base_pv = self._setpoint * closeness + (1.0 - closeness) * (self._setpoint * 0.5)
        noise = random.gauss(0, self._noise_std)
        self._current_pv = 0.8 * self._current_pv + 0.2 * (base_pv + noise)
        process_variable = int(max(0, min(self._process_variable_max, round(self._current_pv))))
        return process_variable


def _make_extra_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mock phase shifter server for testing")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between sending responses (default: 0.0)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=10.0,
        help="Standard deviation of noise added to process variable (default: 10.0)",
    )
    return parser


@experiment(
    experiment_name="phase_shifter_server", 
    extra_parser=_make_extra_parser()
)
def main(context: ExperimentContext) -> None:
    config = context.config
    cli = context.config.cli
    env_args = config.env.args
    host, port = parse_socket_port(env_args.port)

    delay = float(cli.delay)
    noise_std = float(cli.noise_std)

    simulator = PhaseShifterSimulator(
        control_min=env_args.control_min,
        control_max=env_args.control_max,
        setpoint=env_args.setpoint,
        process_variable_max=env_args.process_variable_max,
        noise_std=noise_std,
    )

    context.logger.log(f"Setpoint: {simulator._setpoint}")
    context.logger.log(f"Control range: [{simulator._control_min}, {simulator._control_max}]")
    context.logger.log(f"Process variable max: {simulator._process_variable_max}")
    context.logger.log(f"Optimal control signal: {simulator._optimal_control}")

    def handle_command(command: str, connection: SocketAdapter):
        control = PhaseShifterProtocol.parse_command(command)
        process_variable = simulator.step(control)
        response = PhaseShifterProtocol.format_response(process_variable)
        connection.send(response)

    server = Server(
        host=host,
        port=port,
        command_handler=handle_command,
        delay=delay,
    )
    run_server(server)


if __name__ == "__main__":
    main()
