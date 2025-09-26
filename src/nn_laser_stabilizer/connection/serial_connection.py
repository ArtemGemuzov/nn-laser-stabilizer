import serial

from nn_laser_stabilizer.connection.base_connection import BaseConnection

class SerialConnection(BaseConnection):
    def __init__(self,
                 port: str,
                 timeout: float = 0.1,
                 baudrate: int = 115200,
                 bytesize: int = serial.EIGHTBITS,
                 parity: str = serial.PARITY_NONE,
                 stopbits: int = serial.STOPBITS_ONE):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits

    def open_connection(self):
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=self.bytesize,
                parity=self.parity,
                stopbits=self.stopbits
            )
            if self.serial_connection.is_open:
                print("Serial connection established.")
            else:
                raise ConnectionError("Failed to open serial port.")
        except Exception as ex:
            raise ConnectionError("Error initializing serial connection") from ex

    def close_connection(self):
        if self.serial_connection.is_open:
            self.serial_connection.close()
            print("Serial connection closed.")
        else:
            print("Serial connection already closed.")

    def read_data(self) -> str | None:
        if not self.serial_connection or not self.serial_connection.is_open:
            raise ConnectionError("Serial connection is not open.")
        
        try:
            raw_data = self.serial_connection.readline().decode("utf-8").strip()
            if not raw_data:
                return None
            return raw_data
        except Exception:
            return None
    
    def send_data(self, data_to_send : str):
        if not self.serial_connection or not self.serial_connection.is_open:
            raise ConnectionError("Serial connection is not open.")
    
        data_to_send += '\n'
        try:
            self.serial_connection.write(data_to_send.encode('utf-8'))
        except Exception as ex:
            raise IOError("Error sending data") from ex

