from typing import Optional, Tuple, TypeVar

import torch.multiprocessing as mp
from multiprocessing.connection import Connection

from nn_laser_stabilizer.collector.utils import CollectorCommand, CollectorWorkerErrorInfo

T = TypeVar('T')


class CollectorConnection:
    @staticmethod
    def create_pair() -> Tuple["CollectorConnection", "CollectorConnection"]:
        parent_conn, child_conn = mp.Pipe()
        return CollectorConnection(parent_conn), CollectorConnection(child_conn)
    
    def __init__(self, connection: Connection):
        self._connection = connection
    
    def send_command(self, command: CollectorCommand, data: Optional[T] = None) -> None:
        self._connection.send((command.value, data))
    
    def recv_command(self) -> Tuple[CollectorCommand, Optional[T]]:
        command_str, data = self._connection.recv()
        
        try:
            command = CollectorCommand.from_str(command_str)
        except ValueError as e:
            raise ValueError(f"Unknown command received: {command_str}") from e
        
        return command, data
    
    def poll(self, timeout: Optional[float] = None) -> bool:
        """
        Проверяет, есть ли данные для чтения.
        
        Args:
            timeout: Время ожидания в секундах (None для неблокирующей проверки)
        
        Returns:
            True если есть данные для чтения, False иначе
        """
        return self._connection.poll(timeout if timeout is not None else 0)
    
    def send_worker_ready(self) -> None:
        self.send_command(CollectorCommand.WORKER_READY, None)

    def send_worker_error(self, error: CollectorWorkerErrorInfo) -> None:
        self.send_command(CollectorCommand.WORKER_ERROR, error)
    
    def send_shutdown(self) -> None:
        self.send_command(CollectorCommand.SHUTDOWN, None)

    def send_shutdown_complete(self) -> None:
        self.send_command(CollectorCommand.SHUTDOWN_COMPLETE, None)
    
    def request_weight_update(self) -> None:
        self.send_command(CollectorCommand.REQUEST_WEIGHT_UPDATE, None)
    
    def send_weight_update_done(self) -> None:
        self.send_command(CollectorCommand.WEIGHT_UPDATE_DONE, None)
    
    def wait_for_ready(self, timeout: Optional[float] = None) -> None:
        self._check_timeout(timeout, CollectorCommand.WORKER_READY)
        
        command, data = self.recv_command()
        if command == CollectorCommand.WORKER_READY:
            return
        elif command == CollectorCommand.WORKER_ERROR:
            error: CollectorWorkerErrorInfo = data
            error.raise_exception()
        else:
            self._raise_unexpected_command_error(command, CollectorCommand.WORKER_READY, CollectorCommand.WORKER_ERROR)
    
    def wait_for_weight_update_done(self, timeout: Optional[float] = None) -> None:
        self._check_timeout(timeout, CollectorCommand.WEIGHT_UPDATE_DONE)
        
        command, data = self.recv_command()
        if command == CollectorCommand.WEIGHT_UPDATE_DONE:
            return
        elif command == CollectorCommand.WORKER_ERROR:
            error: CollectorWorkerErrorInfo = data
            error.raise_exception()
        else:
            self._raise_unexpected_command_error(command, CollectorCommand.WEIGHT_UPDATE_DONE, CollectorCommand.WORKER_ERROR)
    
    def wait_for_shutdown_complete(self, timeout: Optional[float] = None) -> None:
        self._check_timeout(timeout, CollectorCommand.SHUTDOWN_COMPLETE)
        
        command, data = self.recv_command()
        if command == CollectorCommand.SHUTDOWN_COMPLETE:
            return
        elif command == CollectorCommand.WORKER_ERROR:
            error: CollectorWorkerErrorInfo = data
            error.raise_exception()
        else:
            self._raise_unexpected_command_error(command, CollectorCommand.SHUTDOWN_COMPLETE, CollectorCommand.WORKER_ERROR)

    def poll_worker_error(self, timeout: Optional[float] = None) -> None:
        """
        Проверяет наличие ошибки от воркера и выбрасывает исключение, если ошибка получена.
        
        Args:
            timeout: Время ожидания в секундах (None для неблокирующей проверки)
        
        Raises:
            RuntimeError: Если получена ошибка от воркера
        """
        if self.poll(timeout):
            command, data = self.recv_command()
            if command == CollectorCommand.WORKER_ERROR:
                error: CollectorWorkerErrorInfo = data
                error.raise_exception()

    def _check_timeout(self, timeout: Optional[float], expected_command: CollectorCommand) -> None:
        if not self.poll(timeout):
            raise RuntimeError(f"Collector process did not send {expected_command} signal within timeout")
    
    def _raise_unexpected_command_error(
        self, 
        received_command: CollectorCommand, 
        *expected_commands: CollectorCommand
    ) -> None:
        expected_str = " or ".join(str(cmd) for cmd in expected_commands)
        raise ValueError(f"Unexpected command received: {received_command}, expected {expected_str}")

