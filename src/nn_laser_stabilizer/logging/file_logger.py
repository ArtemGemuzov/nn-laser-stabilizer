from pathlib import Path


class SimpleFileLogger:
    """
    Простой логгер, который записывает все в один файл одной строкой с переносом.
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / "log.txt"
        self._file_handle = None

        print(f"SimpleFileLogger initialized. Logs: {self.log_file}")

    def log(self, message: str):
        """
        Записывает сообщение в файл.
        """
        # Предполгается однопоточная запись
        if self._file_handle is None:
            self._file_handle = open(self.log_file, 'w', encoding='utf-8')

        self._file_handle.write(message + "\n")

    def close(self):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self):
        try:
            self.close()
        except:
            pass

