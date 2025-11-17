from pathlib import Path
from collections import deque
from threading import Thread
from typing import Optional, Deque
import time


class AsyncFileLogger:
    def __init__(self, log_dir: str, filename: str = "log.log"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / filename
        self._file_handle: Optional[object] = open(self.log_file, 'a', encoding='utf-8')

        # Один логгер на один файл -> используем deque без блокировок
        self._queue: Deque[str] = deque()
        self._stop: bool = False
        self._poll_interval_sec: float = 0.001

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
        self._file_handle.flush()

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

    def close(self) -> None:
        if self._stop:
            return
        self._stop = True
        self._thread.join()
        self._file_handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass