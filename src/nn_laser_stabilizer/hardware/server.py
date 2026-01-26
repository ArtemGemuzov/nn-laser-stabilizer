import socket
import sys
import signal
import time
from typing import Optional, Callable

from nn_laser_stabilizer.hardware.socket import SocketAdapter


class Server: 
    def __init__(
        self,
        host: str,
        port: int,
        command_handler: Callable[[str, SocketAdapter], None],
        delay: float = 0.0,
    ):
        self.host = host
        self.port = port
        self._command_handler = command_handler
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

            print(f"Server listening on {self.host}:{self.port}")
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
                    
                    self._command_handler(command, connection)
                    
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


def run_server(server: Server):
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
