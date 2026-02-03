import argparse
import random
import math

from nn_laser_stabilizer.connection.pid_protocol import PidProtocol
from nn_laser_stabilizer.hardware.socket import parse_socket_port, SocketAdapter
from nn_laser_stabilizer.hardware.server import Server, run_server
from nn_laser_stabilizer.experiment.decorator import experiment
from nn_laser_stabilizer.experiment.context import ExperimentContext


class PidSimulator:
    MAX_DISTANCE = 20.0
    NOISE_STD = 30.0

    def __init__(
        self,
        kp_min: float,
        kp_max: float,
        ki_min: float,
        ki_max: float,
        kd_min: float,
        kd_max: float,
    ):
        self._kp_min = kp_min
        self._kp_max = kp_max
        self._ki_min = ki_min
        self._ki_max = ki_max
        self._kd_min = kd_min
        self._kd_max = kd_max

        self._optimal_kp = self._kp_min + (self._kp_max - self._kp_min) * 0.75
        self._optimal_ki = self._ki_min + (self._ki_max - self._ki_min) * 0.75
        self._optimal_kd = self._kd_min + (self._kd_max - self._kd_min) * 0.75

    def step(
        self,
        kp: float,
        ki: float,
        kd: float,
        control_min: float,
        control_max: float,
        setpoint: int,
    ) -> tuple[int, int]:
        distance = math.sqrt(
            (kp - self._optimal_kp) ** 2
            + (ki - self._optimal_ki) ** 2
            + (kd - self._optimal_kd) ** 2
        )
        closeness = max(0.0, 1.0 - min(distance, self.MAX_DISTANCE) / self.MAX_DISTANCE)
        random_component = random.randint(0, 2000)
        noise = random.gauss(0, self.NOISE_STD)
        process_variable = (
            closeness * setpoint
            + (1.0 - closeness) * random_component
            + noise
        )
        process_variable = int(max(0, min(2000, round(process_variable))))
        control_output = random.randint(int(control_min), int(control_max))
        return process_variable, control_output

    @property
    def optimal_kp(self) -> float:
        return self._optimal_kp

    @property
    def optimal_ki(self) -> float:
        return self._optimal_ki

    @property
    def optimal_kd(self) -> float:
        return self._optimal_kd


def _make_extra_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mock PID server for testing")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between sending responses (default: 0.0)",
    )
    return parser


@experiment(
    experiment_name="pid_server", 
    extra_parser=_make_extra_parser()
)
def main(context: ExperimentContext) -> None:
    config = context.config
    cli = context.config.cli
    env_args = config.env.args
    host, port = parse_socket_port(str(env_args.port))

    delay = float(cli.delay)

    pid_simulator = PidSimulator(
        kp_min=env_args.kp_min,
        kp_max=env_args.kp_max,
        ki_min=env_args.ki_min,
        ki_max=env_args.ki_max,
        kd_min=env_args.kd_min,
        kd_max=env_args.kd_max,
    )

    context.logger.log(
        f"Optimal PID: "
        f"kp={pid_simulator.optimal_kp:.{PidProtocol.KP_DECIMAL_PLACES}f}, "
        f"ki={pid_simulator.optimal_ki:.{PidProtocol.KI_DECIMAL_PLACES}f}, "
        f"kd={pid_simulator.optimal_kd:.{PidProtocol.KD_DECIMAL_PLACES}f}"
    )

    def handle_command(command: str, connection: SocketAdapter):
        kp, ki, kd, control_min, control_max, setpoint = PidProtocol.parse_command(command)
        process_variable, control_output = pid_simulator.step(
            kp=kp,
            ki=ki,
            kd=kd,
            control_min=control_min,
            control_max=control_max,
            setpoint=setpoint,
        )
        response = PidProtocol.format_response(process_variable, control_output)
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
