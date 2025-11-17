from typing import Protocol, Optional


class BaseConnection(Protocol):
    """
    Базовый протокол последовательного соединения.

    Реализации должны обеспечивать открытие/закрытие соединения,
    отправку строковых данных и чтение строкового ответа.
    """

    def open_connection(self) -> None: ...
    def close_connection(self) -> None: ...
    def send_data(self, data_to_send: str) -> None: ...
    def read_data(self) -> Optional[str]: ...


