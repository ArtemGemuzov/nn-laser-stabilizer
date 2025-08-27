# nn-laser-stabilizer
Проект по стабилизации лазера с помощью нейронных сетей.

## Описание
Проект направлен на разработку системы стабилизации лазера с использованием методов машинного обучения.

## Структура репозитория
```text
nn-laser-stabilizer/
├── README.md              # Описание проекта и инструкции
├── requirements.txt       # Список зависимостей проекта
├── pyproject.toml         # Конфигурация пакета
├── src/
│   └── nn_laser_stabilizer/  # Исходный код проекта
├── scripts/               # Исполняемые скрипты
│   └── train_script.py
├── configs/               # Конфигурационные файлы
└── experiments/           # Результаты экспериментов
```

## Установка
1. Клонируйте репозиторий или скачайте zip архив (Code/Download ZIP) и разархивируйте его:
   ```bash
   git clone https://github.com/ArtemGemuzov/nn-laser-stabilizer.git
   ```
1. Перейдите в загруженную директорию:
   ```bash
   cd nn-laser-stabilizer
   ```
1. Создайте окружение в Conda (опционально, если нет действующего)
   ```bash
   conda create -n your_env python=3.10.13
   ```
1. Активируйте свое окружение в Conda:
   ```bash
   conda activate your_env
   ```
1. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
1. Установите пакет:
   ```bash
   pip install -e .
   ```
## Запуск
Запуск скрпита из директории проекта:
```bash
   python scripts/<name_script>.py
```
Выходные данные будут собраны в папке experiments.

## Требования
См. `requirements.txt` для списка необходимых библиотек.