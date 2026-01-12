from typing import Optional
from pathlib import Path
import os


class WorkingDirectoryContext: 
    def __init__(self, target_dir: Path):
        self.target_dir = Path(target_dir)
        self._old_cwd: Optional[Path] = None
    
    def __enter__(self) -> "WorkingDirectoryContext":
        self._old_cwd = Path.cwd()
        os.chdir(self.target_dir)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._old_cwd is not None:
            os.chdir(self._old_cwd)
        return False

