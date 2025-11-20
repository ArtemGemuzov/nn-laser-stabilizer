from pathlib import Path
from typing import Optional, Protocol, Deque
from collections import deque
from threading import Thread
import time

class Logger(Protocol):
    def log(self, message: str) -> None: ...
    def close(self) -> None: ...


class SyncFileLogger:
    def __init__(self, log_dir: str | Path, log_file: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / log_file
        self._file_handle = open(self.log_file, 'a', encoding='utf-8')
        self._closed = False
    
    def log(self, message: str) -> None:
        if self._closed:
            return
        if not message.endswith("\n"):
            message += "\n"
        self._file_handle.write(message)
        self._file_handle.flush()
    
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._file_handle.close()
    
    def __del__(self):
        self.close()


class AsyncFileLogger:
    def __init__(self, log_dir: str | Path, log_file: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / log_file
        self._file_handle: Optional[object] = open(self.log_file, 'a', encoding='utf-8')

        self._queue: Deque[str] = deque()
        self._stop: bool = False
        self._poll_interval_sec: float = 0.01

        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()

    def log(self, message: str) -> None:
        if self._stop:
            return
        self._queue.append(message)

    def _write_line(self, line: str) -> None:
        if not line.endswith("\n"):
            line += "\n"
        self._file_handle.write(line)

    def _worker(self) -> None:
        while True:
            try:
                line = self._queue.popleft()
            except IndexError:
                if self._stop:
                    break
                time.sleep(self._poll_interval_sec)
                continue
            self._write_line(line)
