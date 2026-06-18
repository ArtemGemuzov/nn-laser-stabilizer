"""Адаптеры формата: генерация лог-файлов → канонические таблицы и конфиг.

Разнобой поколений (``collector.jsonl`` / ``env.jsonl`` / keyval / tfevents)
замкнут здесь. Каждый адаптер привязан к каталогу эксперимента и лениво
(``cached_property``) отдаёт канонические таблицы (``step``/``exchange``/
``train``), канонический конфиг и дешёвые метаданные (``present_fields``,
``duration_seconds``). Доменные классы выше по стеку про форматы не знают —
читают только канонические колонки.

Пилот реализует только :class:`CollectorFormat` (поколение neural_controller-v3).
"""

from __future__ import annotations

import re
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from nn_laser_stabilizer.analysis.sources import read_jsonl
from nn_laser_stabilizer.config.config import load_config


_REQUIRED = object()  # сигнал «обязательный ключ» для CanonicalConfig.get


class CanonicalConfig:
    """Канонический срез конфига: единая схема ключей поверх дрейфа поколений.

    Адаптер наполняет его стабильными ключами (``setpoint``, ``error_factor``,
    ``max_delta``, ``exploration.steps`` …); доменные ``Params`` достают из него
    только нужное. Так нет матрицы «адаптер × домен».
    """

    def __init__(self, values: dict[str, Any]):
        self._values = values

    def get(self, key: str, default: Any = _REQUIRED) -> Any:
        if key in self._values:
            return self._values[key]
        if default is _REQUIRED:
            raise KeyError(f"В каноническом конфиге нет обязательного ключа '{key}'")
        return default


def _first_scalar(value: Any) -> float:
    """Достать скаляр из значения-списка ``[x]`` (поля политики логируются так)."""
    if isinstance(value, (list, tuple)) and value:
        return float(value[0])
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


# Таймстемп в начале строки console.log: "[2026-06-09 17:49:47,123] ..."
_TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:,\d{3})?\]")


def _read_duration_seconds(log_path: Path) -> float:
    """Длительность прогона по первому и последнему таймстемпу console.log."""
    first = last = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _TS_RE.match(line)
            if not m:
                continue
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            if first is None:
                first = t
            last = t
    if first is None or last is None:
        return float("nan")
    return (last - first).total_seconds()


class CollectorFormat:
    """Поколение neural_controller-v3: ``collector.jsonl`` + ``train.jsonl``.

    ``collector.jsonl`` несёт два потока событий в одном файле: ``step``
    (env_info + policy_info) и ``exchange`` (обмен с железом). Сырой парс
    кешируется, поэтому ``step`` и ``exchange`` не парсят файл дважды.
    """

    # raw-поле (после flatten в read_jsonl) → каноническая колонка.
    # Поля политики (списки [x]) помечены для извлечения скаляра.
    _STEP_RENAME = {
        "process_variable": "process_variable",
        "control_output": "control_output",
        "error": "error",
        "prev_error": "prev_error",
        "prev_prev_error": "prev_prev_error",
        "prev_control_output": "prev_control_output",
        "prev_prev_control_output": "prev_prev_control_output",
        "setpoint": "setpoint",
        "action_norm": "action_norm",
        "reward": "reward",
        "reward_error_term": "reward_error_term",
        "cost_action": "cost_action",
        "cost_control": "cost_control",
        "cost_barrier": "cost_barrier",
        "cost_terminal": "cost_terminal",
        "reward_alive": "reward_alive",
        "terminated": "terminated",
        "truncated": "truncated",
        "step": "episode_step",
        "step_interval_us": "step_interval_us",
        "policy_mean_action": "policy_mean",
        "policy_std": "policy_std",
        "policy_log_prob": "policy_log_prob",
        "policy_raw_action": "policy_raw",
        "policy_policy_mode": "policy_mode",
    }
    _STEP_SCALAR_COLS = (
        "policy_mean", "policy_std", "policy_log_prob", "policy_raw",
    )
    _EXCHANGE_RENAME = {
        "control_output": "control_output",
        "process_variable": "process_variable",
    }
    _TRAIN_RENAME = {
        "step": "train_step",
        "actor_loss": "actor_loss",
        "loss_q1": "q1_loss",
        "loss_q2": "q2_loss",
        "alpha": "alpha",
        "alpha_loss": "alpha_loss",
        "entropy": "entropy",
        "buffer_size": "buffer_size",
    }
    _EVAL_RENAME = {
        "step": "step",
        "episodes": "episodes",
        "reward_mean": "reward_mean",
        "reward_sum": "reward_sum",
        "reward_max": "reward_max",
        "reward_min": "reward_min",
    }

    def __init__(self, dir: Path):
        self._dir = Path(dir)

    @staticmethod
    def matches(dir: Path) -> bool:
        return (Path(dir) / "collector.jsonl").exists()

    # --- сырой парс (кешируется, один на step+exchange) ---
    @cached_property
    def _raw(self) -> pd.DataFrame:
        return read_jsonl(self._dir / "collector.jsonl")

    def _canon(
        self, df: pd.DataFrame, rename: dict[str, str], scalar_cols: tuple[str, ...] = ()
    ) -> pd.DataFrame:
        present = {raw: canon for raw, canon in rename.items() if raw in df.columns}
        out = df[list(present)].rename(columns=present).reset_index(drop=True)
        for col in scalar_cols:
            if col in out.columns:
                out[col] = out[col].map(_first_scalar)
        # global_step — позиционный счётчик (в логе его нет; блокнот так и делает)
        out.index = pd.RangeIndex(1, len(out) + 1, name="global_step")
        return out

    @cached_property
    def step(self) -> pd.DataFrame:
        rows = self._raw[self._raw["event"] == "step"]
        return self._canon(rows, self._STEP_RENAME, self._STEP_SCALAR_COLS)

    @cached_property
    def exchange(self) -> pd.DataFrame:
        rows = self._raw[self._raw["event"] == "exchange"]
        return self._canon(rows, self._EXCHANGE_RENAME)

    @cached_property
    def _train_raw(self) -> pd.DataFrame:
        # train.jsonl смешивает event=step (метрики) и event=evaluation (награды)
        return read_jsonl(self._dir / "train.jsonl")

    def _train_subset(self, event: str, rename: dict[str, str]) -> pd.DataFrame:
        raw = self._train_raw
        rows = raw[raw["event"] == event] if "event" in raw.columns else raw
        return self._canon(rows, rename)

    @cached_property
    def train(self) -> pd.DataFrame:
        return self._train_subset("step", self._TRAIN_RENAME)

    @cached_property
    def evaluation(self) -> pd.DataFrame:
        return self._train_subset("evaluation", self._EVAL_RENAME)

    @cached_property
    def raw_config(self):
        """Полный сырой Config (для реконструкции агента: algorithm/env/seed)."""
        return load_config(self._dir / "config.yaml")

    @cached_property
    def config(self) -> CanonicalConfig:
        c = self.raw_config
        args = c.env.args
        # target_entropy: путь в конфиге, иначе значение по умолчанию −1
        target_entropy = c.get("algorithm.target_entropy_value", -1)
        gamma = c.get("algorithm.gamma")
        return CanonicalConfig({
            "setpoint": float(args.setpoint) / 10,
            "error_factor": float(args.get("error_normalixation_factor")),
            "max_delta": float(args.action.max_delta),
            "control_min": float(args.control_min),
            "control_max": float(args.control_max),
            # exploration есть не всегда (например, в инференсе) → по умолчанию 0
            "exploration.steps": int(c.get("exploration.steps", 0)),
            "target_entropy": float(target_entropy),
            "gamma": float(gamma) if gamma is not None else None,
            # флаги состава вектора наблюдений (лог пишет все каналы, а в вектор
            # входят только отмеченные observe_*) — по ним строится observations.names
            "observe_prev_error": bool(args.get("observe_prev_error", False)),
            "observe_prev_prev_error": bool(args.get("observe_prev_prev_error", False)),
            "observe_prev_control_output": bool(args.get("observe_prev_control_output", False)),
            "observe_prev_prev_control_output": bool(args.get("observe_prev_prev_control_output", False)),
        })

    @cached_property
    def duration_seconds(self) -> float:
        return _read_duration_seconds(self._dir / "console.log")


_ADAPTERS = (CollectorFormat,)


def detect_format(dir: Path):
    """Выбрать адаптер поколения по наличию характерных файлов."""
    dir = Path(dir)
    for adapter in _ADAPTERS:
        if adapter.matches(dir):
            return adapter(dir)
    raise ValueError(f"Не удалось определить формат эксперимента: {dir}")
