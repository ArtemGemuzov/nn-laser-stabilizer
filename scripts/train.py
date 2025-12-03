from pathlib import Path
import runpy


if __name__ == "__main__":
    this_dir = Path(__file__).resolve().parent
    target = this_dir / "train_online.py"

    if not target.exists():
        raise FileNotFoundError(f"train_online.py not found at {target}")

    runpy.run_path(str(target), run_name="__main__")