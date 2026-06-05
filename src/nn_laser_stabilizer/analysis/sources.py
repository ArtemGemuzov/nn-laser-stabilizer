"""Чтение текстовых логов экспериментов в :class:`pandas.DataFrame`.

Форматы логов проекта менялись от поколения к поколению экспериментов:
разные префиксы (``[ENV]``, ``[TRAIN]``, ``[PID]``, ``[PHASE_SHIFTER]`` или
вовсе без префикса), разные наборы полей и даже единицы измерения в
значениях (``step_interval=0.0us``). Поэтому вместо фиксированных regex под
конкретную схему здесь один общий парсер строк вида::

    [TAG] event: key=value key=value ...

Он извлекает все пары ``key=value``, какие есть в строке, приводит типы и
возвращает DataFrame с объединением присутствующих колонок. Тип записи
(``tag`` и ``event``) попадает в отдельные колонки, чтобы строки разных
схем в одном файле можно было отфильтровать уже средствами pandas.

Имена колонок сохраняются как в логе (snake_case), без переименования —
приведение к «каноническим» именам и подписи для графиков относятся к
более высокому слою и здесь не делаются.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

try:  # orjson быстрее на больших файлах (есть в requirements-dev), иначе stdlib
    import orjson as _json
except ImportError:  # pragma: no cover
    import json as _json


def _json_loads(line: str):
    return _json.loads(line)


PathLike = Union[str, Path]

# Необязательный префикс "[TAG]" и необязательный "event:" в начале строки.
# event сопоставляется только если за словом сразу идёт двоеточие, поэтому
# строки вида "step=0 Loss/Critic=..." (без verb) не дают ложного event.
_PREFIX_RE = re.compile(
    r"^\s*"
    r"(?:\[(?P<tag>[^\]]+)\]\s*)?"
    r"(?:(?P<event>[A-Za-z_]+):)?"
)

# Ключ может содержать '/' (legacy: Loss/Critic) и '_'. Значение — всё до пробела.
_TOKEN_RE = re.compile(r"([\w/]+)=(\S+)")

# Единицы измерения, которые встречаются в значениях (step_interval=0.0us).
_UNIT_RE = re.compile(r"^(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:us|ms|ns|s)$")


def _coerce(value: str):
    """Привести строковое значение к int / float / bool, иначе оставить строкой."""
    unit_match = _UNIT_RE.match(value)
    if unit_match:
        value = unit_match.group(1)

    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"

    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_keyval_log(
    path: PathLike,
    *,
    event: Optional[str] = None,
    tag: Optional[str] = None,
) -> pd.DataFrame:
    """Разобрать лог из строк ``[TAG] event: key=value ...`` в DataFrame.

    Каждая строка, содержащая хотя бы одну пару ``key=value``, становится
    строкой DataFrame; колонки — объединение всех встреченных ключей. Тип
    записи добавляется колонками ``tag`` и ``event`` (если присутствует
    хотя бы в одной строке файла).

    Args:
        path: путь к лог-файлу.
        event: если задан, оставить только строки с этим verb (без учёта
            регистра), напр. ``"step"``, ``"send"``, ``"read"``.
        tag: если задан, оставить только строки с этим префиксом ``[TAG]``
            (без учёта регистра), напр. ``"ENV"``, ``"PHASE_SHIFTER"``.

    Returns:
        DataFrame с распарсенными полями (пустой, если совпадений нет).
    """
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            tokens = _TOKEN_RE.findall(line)
            if not tokens:
                continue

            prefix = _PREFIX_RE.match(line)
            row_tag = prefix.group("tag") if prefix else None
            row_event = prefix.group("event") if prefix else None

            if event is not None and (row_event or "").lower() != event.lower():
                continue
            if tag is not None and (row_tag or "").lower() != tag.lower():
                continue

            row = {key: _coerce(val) for key, val in tokens}
            row["tag"] = row_tag
            row["event"] = row_event
            rows.append(row)

    df = pd.DataFrame(rows)
    # Убрать служебные колонки, если они пусты во всех строках.
    for meta in ("tag", "event"):
        if meta in df.columns and df[meta].isna().all():
            df = df.drop(columns=meta)
    return df


def parse_env_log(path: PathLike) -> pd.DataFrame:
    """Прочитать ``env_logs/env.log`` (наблюдения среды, шаги эпизодов).

    Возвращает все записи лога. Разные поколения пишут разный набор
    полей; при необходимости фильтруйте по колонке ``event`` (напр.
    ``df[df["event"] == "step"]``).
    """
    return parse_keyval_log(path)


def parse_train_log(path: PathLike) -> pd.DataFrame:
    """Прочитать ``train_logs/train.log`` (метрики обучения, лоссы).

    Поля различаются между поколениями (``Loss/Critic``/``Loss/Actor`` в
    старом формате, ``loss_q1``/``loss_q2``/``actor_loss`` в новом) и даже
    между строками одного файла (актор обновляется не на каждом шаге).
    """
    return parse_keyval_log(path)


def parse_connection_log(path: PathLike) -> pd.DataFrame:
    """Прочитать лог соединения (отправки/чтения контроллера).

    Содержит записи разных типов (``send``/``read``, в старом формате —
    ``SEND``/``READ``/``OPEN``/``CLOSE``); тип записи доступен в колонке
    ``event``.
    """
    return parse_keyval_log(path)


def read_tfevents(logdir: PathLike) -> pd.DataFrame:
    """Прочитать скаляры из TensorBoard ``tfevents`` в широкий DataFrame.

    Используется для самого старого поколения экспериментов, где метрики
    писались не в текст, а в ``events.out.tfevents.*`` (директории
    ``train_logs``/``env_logs``). Возвращает DataFrame с колонкой ``step``
    и по колонке на каждый скалярный тег (``Loss/Critic``,
    ``Observation/x`` и т.п.).

    ВАЖНО: ``EventAccumulator`` по умолчанию прореживает скаляры до 10000
    точек на тег. Здесь ``size_guidance`` для скаляров выставлен в ``0``
    (без лимита), чтобы вернуть ВСЕ записанные точки.

    Зависимость ``tensorboard`` импортируется лениво, чтобы чтение
    текстовых/JSONL логов работало без неё.
    """
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(
        str(logdir),
        size_guidance={event_accumulator.SCALARS: 0},  # 0 = загрузить все точки
    )
    ea.Reload()

    rows = []
    for tag in ea.Tags()["scalars"]:
        for e in ea.Scalars(tag):
            rows.append(
                {
                    "wall_time": e.wall_time,
                    "step": e.step,
                    "tag": tag,
                    "value": e.value,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # pivot_table (а не pivot): переживает дубликаты (step, tag) после
    # рестартов обучения; "last" берёт последнее записанное значение шага.
    return df.pivot_table(
        index="step", columns="tag", values="value", aggfunc="last"
    ).reset_index()


def _flatten_dict(data: dict, prefix: str = "") -> dict:
    """Рекурсивно расплющить вложенный dict в плоский (ключи через точку)."""
    flat: dict = {}
    stack = [(prefix, data)]
    while stack:
        pref, cur = stack.pop()
        for k, v in cur.items():
            key = pref + k
            if isinstance(v, dict):
                stack.append((key + ".", v))
            else:
                flat[key] = v
    return flat


def _flatten_record(record: dict) -> dict:
    """Привести запись JSONL к плоскому виду.

    Новый формат ``collector.jsonl`` несёт вложенные ``env_info`` и
    ``policy_info``; поля окружения раскрываются плоско (``error``,
    ``reward``, …), поля политики — с префиксом ``policy_``. Плоские
    записи (``train.jsonl``, ``pid_data.jsonl``, старый ``env.jsonl``)
    возвращаются как есть.
    """
    if "env_info" not in record and "policy_info" not in record:
        return record

    out = {
        k: v for k, v in record.items() if k not in ("env_info", "policy_info")
    }
    env_info = record.get("env_info")
    if isinstance(env_info, dict):
        out.update(_flatten_dict(env_info))
    policy_info = record.get("policy_info")
    if isinstance(policy_info, dict):
        out.update(_flatten_dict(policy_info, prefix="policy_"))
    return out


def read_jsonl(
    path: PathLike,
    *,
    event: Optional[str] = None,
    flatten: bool = True,
) -> pd.DataFrame:
    """Прочитать JSONL-лог (по одному JSON-объекту на строку) в DataFrame.

    Это самый массовый формат (поколение neural_controller-v1/v2/v3 и
    run-pid): ``collector.jsonl``/``env.jsonl`` (обмен с железом),
    ``train.jsonl`` (метрики обучения), ``pid_data.jsonl`` (трасса PID).
    Отсутствующие в части строк ключи становятся ``NaN``; поля-дискриминаторы
    ``source``/``event`` (если есть) остаются обычными колонками — по смыслу
    они те же, что ``tag``/``event`` у :func:`parse_keyval_log`, что даёт
    единообразие для канон-слоя.

    Args:
        path: путь к ``.jsonl`` файлу.
        event: если задан, оставить только записи с этим ``event``
            (напр. ``"step"`` для train.jsonl, ``"exchange"`` для collector).
        flatten: расплющивать вложенные ``env_info``/``policy_info`` в
            плоские колонки (поля политики — с префиксом ``policy_``).
            При ``True`` адаптер всегда возвращает плоский DataFrame, как и
            :func:`parse_keyval_log`/:func:`read_tfevents`. Плоские схемы
            не затрагиваются.

    Returns:
        DataFrame с объединением всех встреченных ключей.
    """
    if not flatten:
        df = pd.read_json(path, lines=True)
    else:
        rows = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = _json_loads(line)
                except ValueError:
                    continue
                rows.append(_flatten_record(record))
        df = pd.DataFrame(rows)

    if event is not None and "event" in df.columns:
        df = df[df["event"] == event].reset_index(drop=True)
    return df
