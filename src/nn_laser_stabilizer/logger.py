from pathlib import Path
from typing import Protocol, Deque, Any
from collections import deque
from threading import Thread
from multiprocessing import Process, Queue, Event
from queue import Empty, Full
from datetime import datetime
import time
import re


class Logger(Protocol):
    def log(self, message: str) -> None: ...
    def close(self) -> None: ...


class PrefixedLogger:
    def __init__(self, logger: Logger, prefix: str):
        self._logger = logger
        self._prefix = prefix
    
    def log(self, message: str) -> None:
        self._logger.log(f"[{self._prefix}] {message}")
    
    def close(self) -> None:
        pass


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


class ConsoleLogger:
    def __init__(self, log_dir: str | Path, log_file: str):
        self._file_logger = SyncFileLogger(log_dir=log_dir, log_file=log_file)
    
    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message, end='' if message.endswith("\n") else '\n')
        self._file_logger.log(formatted_message)
    
    def close(self) -> None:
        self._file_logger.close()
    
    def __del__(self):
        self.close()


# TODO: сейчас класс непотокобезопасный: возможно закрытие файла во время записи
class AsyncFileLogger:
    POLL_INTERVAL_SEC = 0.1

    def __init__(self, log_dir: str | Path, log_file: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / log_file
        self._file_handle = open(self.log_file, 'a', encoding='utf-8')

        self._queue: Deque[str] = deque()
        self._stop: bool = False

        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()

    def log(self, message: str) -> None:
        if self._stop:
            return
        self._queue.append(message)

    def _write_line(self, line: str) -> None:
        line += "\n"
        self._file_handle.write(line)

    def _worker(self) -> None:
        while True:
            if self._stop and not self._queue:
                break

            try:
                line = self._queue.popleft()
                self._write_line(line)
            except IndexError:
                time.sleep(self.POLL_INTERVAL_SEC)

    
    def close(self) -> None:
        if self._stop:
            return
        self._stop = True

        self._thread.join()
        self._file_handle.close()
    
    def __del__(self):
        self.close()


class ProcessFileLogger:
    def __init__(self, log_dir: str | Path, log_file: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / log_file
        
        self._queue: Queue[str] = Queue(maxsize=10_000) 

        self._stop_event = Event()
        self._stop = False
        
        self._process = Process(
            target=self._worker,
            args=(self._queue, self.log_file, self._stop_event),
            daemon=False
        )
        self._process.start()
    
    def log(self, message: str) -> None:
        if self._stop:
            return
        
        try:
            self._queue.put_nowait(message)  
        except Full:
            pass
    
    @staticmethod
    def _worker(queue: Queue, log_file: Path, stop_event) -> None:    
        with open(log_file, 'a', encoding='utf-8') as file:
            while True:
                if stop_event.is_set():
                    file.flush()
                    break

                try:
                    message: str = queue.get(timeout=0.1)
                except Empty:
                    continue

                
                message += "\n"
                file.write(message)
                  
    
    def close(self) -> None:
        if self._stop:
            return
        self._stop = True

        self._stop_event.set()  

        self._process.join(timeout=1.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=0.5)
    
    def __del__(self):
        self.close()


def format_log_entry(prefix: str, event_type: str, **kwargs: Any) -> str:
    if not kwargs:
        return f"[{prefix}] {event_type}:"
    parts = [f"{k}={v}" for k, v in kwargs.items()]
    return f"[{prefix}] {event_type}: {' '.join(parts)}"


def parse_log_entry(line: str) -> tuple[str, str, dict[str, str]]:
    pattern = re.compile(
        r"\[(?P<prefix>[^\]]+)\]\s+"
        r"(?P<event_type>\w+):\s+"
        r"(?P<params>.*)"
    )
    
    match = pattern.match(line.strip())
    if not match:
        raise ValueError(f"Invalid log format: {line!r}")
    
    prefix = match.group('prefix')
    event_type = match.group('event_type')
    params_str = match.group('params')
    
    params = {}
    param_pattern = re.compile(r"(\w+)=([^\s]+)")
    for param_match in param_pattern.finditer(params_str):
        key = param_match.group(1)
        value = param_match.group(2)
        params[key] = value
    
    return prefix, event_type, params