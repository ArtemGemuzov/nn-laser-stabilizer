# nn-laser-stabilizer
Проект по стабилизации лазера с помощью нейронных сетней.

## Описание
Проект направлен на разработку системы стабилизации лазера с использованием методов машинного обучения и нейронных сетей.

## Структура репозитория
```text
nn-laser-stabilizer/
├── README.md              # Описание проекта и инструкции
├── requirements.txt       # Список зависимостей проекта
├── pyproject.toml         # Конфигурация пакета
├── src/
│   └── nn_laser_stabilizer/  # Исходный код проекта
│       ├── __init__.py
│       ├── model.py
│       ├── train.py
│       └── utils.py
├── scripts/               # Исполняемые скрипты
│   └── train_script.py
├── configs/               # Конфигурационные файлы
└── experiments/           # Результаты экспериментов
```

## Установка
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/ArtemGemuzov/nn-laser-stabilizer.git
   ```
1. Перейдите в загруженную директорию:
   ```bash
      cd nn-laser-stabilizer
   ```
1. Активируйте свое окружение в Conda:
   ```bash
   conda activate your_env
   ```
1. Установите пакет:
   ```bash
   pip install -e .
   ```
1. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
## Запуск
Запуск обучения модели:
```bash
   python scripts/name_script.py
```

## Требования
См. `requirements.txt` для списка необходимых библиотек.