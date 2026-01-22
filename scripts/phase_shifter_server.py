import socket
import sys
import argparse
import random
import signal
import math
import time
from typing import Optional

from nn_laser_stabilizer.experiment.config import load_config, find_config_path
from nn_laser_stabilizer.connection.phase_shifter_protocol import PhaseShifterProtocol
from nn_laser_stabilizer.hardware.socket import parse_socket_port, SocketAdapter


class PhaseShifterSimulator:
    """Симулятор фазовращателя для тестирования окружения."""
    
    def __init__(
        self,
        control_min: int,
        control_max: int,
        setpoint: int,
        noise_std: float = 10.0,
    ):
        self._control_min = control_min
        self._control_max = control_max
        self._setpoint = setpoint
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
        process_variable = round(self._current_pv)
        return process_variable


class PhaseShifterServer:
    def __init__(
        self,
        host: str,
        port: int,
        simulator: PhaseShifterSimulator,
        delay: float = 0.0,
    ):
        self.host = host
        self.port = port
        self._simulator = simulator
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

            print(f"Phase shifter server listening on {self.host}:{self.port}")
            print(f"Setpoint: {self._simulator._setpoint}")
            print(f"Control range: [{self._simulator._control_min}, {self._simulator._control_max}]")
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
                    
                    control = PhaseShifterProtocol.parse_command(command)
            
                    process_variable = self._simulator.step(control)
                    
                    response = PhaseShifterProtocol.format_response(process_variable)
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
    parser = argparse.ArgumentParser(description="Mock phase shifter server for testing")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., 'neural_pid')"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between sending responses (default: 0.0)"
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=10.0,
        help="Standard deviation of noise added to process variable (default: 10.0)"
    )
    
    args = parser.parse_args()
    
    config_path = find_config_path(args.config)
    config = load_config(config_path)
    
    env_args = config.env.args
    host, port = parse_socket_port(env_args.port)
    
    simulator = PhaseShifterSimulator(
        control_min=env_args.control_min,
        control_max=env_args.control_max,
        setpoint=env_args.setpoint,
        noise_std=args.noise_std,
    )
    
    server = PhaseShifterServer(
        host=host,
        port=port,
        simulator=simulator,
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
