"""
Скрипт для релиза новой версии пакета.

Использование:
    python scripts/release.py major   # Увеличивает major версию (5.0.4 -> 6.0.0)
    python scripts/release.py minor   # Увеличивает minor версию (5.0.4 -> 5.1.0)
    python scripts/release.py patch   # Увеличивает patch версию (5.0.4 -> 5.0.5)
"""

import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path

MONTHS_RU = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля",
    5: "мая", 6: "июня", 7: "июля", 8: "августа",
    9: "сентября", 10: "октября", 11: "ноября", 12: "декабря"
}


def get_current_version(pyproject_path: Path) -> str:
    content = pyproject_path.read_text(encoding='utf-8')
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError("Не удалось найти версию в pyproject.toml")
    return match.group(1)


def increment_version(version: str, version_type: str) -> str:
    parts = list(map(int, version.split('.')))
    
    if len(parts) != 3:
        raise ValueError(f"Неверный формат версии: {version}. Ожидается формат X.Y.Z")
    
    if version_type == "major":
        parts[0] += 1
        parts[1] = 0
        parts[2] = 0
    elif version_type == "minor":
        parts[1] += 1
        parts[2] = 0
    elif version_type == "patch":
        parts[2] += 1
    else:
        raise ValueError(f"Неизвестный тип версии: {version_type}. Используйте major, minor или patch")
    
    return ".".join(map(str, parts))


def update_pyproject_version(pyproject_path: Path, new_version: str) -> None:
    content = pyproject_path.read_text(encoding='utf-8')
    content = re.sub(
        r'version\s*=\s*"[^"]+"',
        f'version = "{new_version}"',
        content
    )
    pyproject_path.write_text(content, encoding='utf-8')
    print(f"✓ Версия в pyproject.toml обновлена до {new_version}")
    
    subprocess.run(
        ["git", "add", str(pyproject_path)],
        check=True
    )
    subprocess.run(
        ["git", "commit", "-m", f"Поднял версию пакета до {new_version}"],
        check=True
    )
    print(f"✓ Изменения закоммичены")


def get_release_message(version: str) -> str:
    """Формирует сообщение для тега в формате 'Релиз версии v5.0.4 от 16 января 2026 г.'"""
    now = datetime.now()
    day = now.day
    month = MONTHS_RU[now.month]
    year = now.year
    return f"Релиз версии v{version} от {day} {month} {year} г."


def check_git_status() -> None:
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError:
        raise RuntimeError("Не найден git репозиторий")
    
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True
    )
    
    if result.stdout.strip():
        changes = result.stdout.strip()
        raise RuntimeError(
            f"Нельзя создать релиз при наличии незакоммиченных изменений. "
            f"Сначала закоммитьте или отмените изменения.\n"
            f"Обнаруженные изменения:\n{changes}"
        )


def create_and_push_tag(version: str, message: str) -> None:
    tag_name = f"v{version}"
    
    result = subprocess.run(
        ["git", "tag", "-l", tag_name],
        capture_output=True,
        text=True,
        check=True
    )
    
    if result.stdout.strip():
        raise RuntimeError(f"Тег {tag_name} уже существует")
    
    subprocess.run(
        ["git", "tag", "-a", tag_name, "-m", message],
        check=True
    )
    print(f"✓ Тег {tag_name} создан")
    
    subprocess.run(
        ["git", "push", "origin", tag_name],
        check=True
    )
    print(f"✓ Тег {tag_name} опубликован")


def main():
    parser = argparse.ArgumentParser(
        description="Автоматизация процесса релиза пакета"
    )
    parser.add_argument(
        "version_type",
        choices=["major", "minor", "patch"],
        help="Тип версии для увеличения: major, minor или patch"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    if not pyproject_path.exists():
        raise FileNotFoundError(f"Файл {pyproject_path} не найден")
    
    current_version = get_current_version(pyproject_path)
    print(f"Текущая версия: {current_version}")
    
    new_version = increment_version(current_version, args.version_type)
    print(f"Новая версия: {new_version}")
    
    check_git_status()
    
    update_pyproject_version(pyproject_path, new_version)
    
    release_message = get_release_message(new_version)
    print(f"Сообщение тега: {release_message}")
    
    create_and_push_tag(new_version, release_message)
    
    print(f"\n✓ Релиз версии {new_version} успешно создан и опубликован!")


if __name__ == "__main__":
    main()
