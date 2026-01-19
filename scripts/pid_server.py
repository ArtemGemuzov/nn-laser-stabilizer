import socket
import sys
import argparse
import random
import signal
import math
import time
from typing import Optional

from nn_laser_stabilizer.experiment.config import load_config, find_config_path
from nn_laser_stabilizer.connection.pid_protocol import PidProtocol
from nn_laser_stabilizer.hardware.socket import parse_socket_port, SocketAdapter


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
            (kp - self._optimal_kp) ** 2 + 
            (ki - self._optimal_ki) ** 2 +
            (kd - self._optimal_kd) ** 2
        )
        
        closeness = max(0.0, 1.0 - min(distance, self.MAX_DISTANCE) / self.MAX_DISTANCE)
        
        random_component = random.randint(0, 2000)
        noise = random.gauss(0, self.NOISE_STD)
        process_variable = (
            closeness * setpoint + 
            (1.0 - closeness) * random_component + 
            noise
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


class PidServer:
    def __init__(
        self,
        host: str,
        port: int,
        pid_simulator: PidSimulator,
        delay: float = 0.0,
    ):
        self.host = host
        self.port = port
        self._pid_simulator = pid_simulator
        self._delay = delay
        self.socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.running = False
        
    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)

            print(f"Mock PID server listening on {self.host}:{self.port}")
            print(
                f"Optimal PID: "
                f"kp={self._pid_simulator.optimal_kp:.{PidProtocol.KP_DECIMAL_PLACES}f}, "
                f"ki={self._pid_simulator.optimal_ki:.{PidProtocol.KI_DECIMAL_PLACES}f}, "
                f"kd={self._pid_simulator.optimal_kd:.{PidProtocol.KD_DECIMAL_PLACES}f}"
            )
            print("Waiting for connection...")
            
            self.running = True
            self.client_socket, address = self.socket.accept()
            print(f"Client connected from {address}")
            self._handle_client()
        finally:
            self._cleanup()
    
    
    def _handle_client(self):
        assert self.client_socket is not None
        connection = SocketAdapter(self.client_socket)
        
        try:
            while self.running:
                try:
                    command = connection.read()
                    if not command:
                        continue
                    
                    kp, ki, kd, control_min, control_max, setpoint = PidProtocol.parse_command(command)
            
                    process_variable, control_output = self._pid_simulator.step(
                        kp=kp,
                        ki=ki,
                        kd=kd,
                        control_min=control_min,
                        control_max=control_max,
                        setpoint=setpoint,
                    )
                    
                    response = PidProtocol.format_response(process_variable, control_output)
                    connection.send(response)
                    
                    if self._delay > 0:
                        time.sleep(self._delay)
                    
                except ConnectionError as e:
                    print("Client disconnected")
                    break
                except Exception as e:
                    print(f"Error handling client: {e}")
                    break
                    
        except Exception as e:
            print(f"Error in client handler: {e}")
        finally:
            connection.close()
            self.client_socket = None
    
    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()
    
    def _cleanup(self):
        if self.client_socket:
            self.client_socket.close()
        if self.socket:
            self.socket.close()
        print("Server stopped")


def main():
    parser = argparse.ArgumentParser(description="Mock PID server for testing")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., 'pid_delta_tuning')"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between sending responses (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    config_path = find_config_path(args.config)
    config = load_config(config_path)
    
    env_args = config.env.args
    host, port = parse_socket_port(str(env_args.port))
    
    pid_simulator = PidSimulator(
        kp_min=env_args.kp_min,
        kp_max=env_args.kp_max,
        ki_min=env_args.ki_min,
        ki_max=env_args.ki_max,
        kd_min=env_args.kd_min,
        kd_max=env_args.kd_max,
    )
    
    server = PidServer(
        host=host,
        port=port,
        pid_simulator=pid_simulator,
        delay=args.delay,
    )
    
    def signal_handler(sig, frame):
        print("\nShutting down server...")
        server.stop()
        sys.exit()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        server.start()
    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        print(f"Server error: {e}")
        server.stop()


if __name__ == "__main__":
    main()

