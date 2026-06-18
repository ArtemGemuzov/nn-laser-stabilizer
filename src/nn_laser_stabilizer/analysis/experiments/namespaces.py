"""Неймспейсы — тонкие типизированные вьюхи над каноническими таблицами.

Наружу отдают именованные доменные ``Series`` (и скаляры/точки), а не
``DataFrame`` и не имена колонок. Логики чтения здесь нет — только выборка
канонических колонок и производные величины (денормализация, время, фазы).
Конвенция имён: голое имя — физические единицы, суффикс ``_norm`` — нормированное.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd


# --- время ---

class Time:
    """Время оси (приближённый пересчёт global_step → длительность), с единицами."""

    def __init__(self, index: pd.Index, duration_seconds: float):
        gs = index.to_series().astype(float)
        self._seconds = gs / gs.max() * duration_seconds

    @property
    def seconds(self) -> pd.Series:
        return self._seconds

    @property
    def minutes(self) -> pd.Series:
        return self._seconds / 60.0

    @property
    def hours(self) -> pd.Series:
        return self._seconds / 3600.0


class TimePoint:
    """Точка во времени (граница фазы) с единицами; ``steps`` точна, время — нет."""

    def __init__(self, step: int, max_step: int, duration_seconds: float):
        self._step = step
        self._seconds = (step / max_step * duration_seconds) if max_step else float("nan")

    @property
    def steps(self) -> int:
        return self._step

    @property
    def seconds(self) -> float:
        return self._seconds

    @property
    def minutes(self) -> float:
        return self._seconds / 60.0

    @property
    def hours(self) -> float:
        return self._seconds / 3600.0


class StepInterval:
    """Реальный интервал между шагами (диагностика COM), с единицами."""

    def __init__(self, df: pd.DataFrame):
        self._us = df["step_interval_us"]

    @property
    def microseconds(self) -> pd.Series:
        return self._us

    @property
    def milliseconds(self) -> pd.Series:
        return self._us / 1e3

    @property
    def seconds(self) -> pd.Series:
        return self._us / 1e6


# --- фазы ---

def _contiguous_ranges(mask: pd.Series) -> list[tuple[int, int]]:
    """Непрерывные диапазоны (start_gs, end_gs) по индексу там, где маска True."""
    ranges: list[tuple[int, int]] = []
    start = prev = None
    for gs, flag in zip(mask.index, mask.to_numpy()):
        if flag and start is None:
            start = gs
        if not flag and start is not None:
            ranges.append((start, prev))
            start = None
        prev = gs
    if start is not None and prev is not None:
        ranges.append((start, prev))
    return ranges


class Phase:
    """Фаза прогона: маска по шагам (точная), диапазоны, и границы как точки."""

    def __init__(self, mask: pd.Series, max_step: int, duration_seconds: float):
        self._mask = mask
        self._max_step = max_step
        self._duration = duration_seconds

    @property
    def mask(self) -> pd.Series:
        return self._mask

    @cached_property
    def ranges(self) -> list[tuple[int, int]]:
        return _contiguous_ranges(self._mask)

    @property
    def start(self) -> TimePoint | None:
        r = self.ranges
        return TimePoint(r[0][0], self._max_step, self._duration) if r else None

    @property
    def end(self) -> TimePoint | None:
        r = self.ranges
        return TimePoint(r[-1][1], self._max_step, self._duration) if r else None


class Phases:
    """Временна́я структура прогона: фазы ПИД / НС / оценки."""

    def __init__(self, df: pd.DataFrame, pid_steps: int, duration_seconds: float):
        gs = df.index.to_series()
        max_step = int(gs.max())
        pid_mask = gs <= pid_steps
        if "policy_mode" in df.columns:
            eval_mask = df["policy_mode"].eq("eval").fillna(False)
        else:
            eval_mask = pd.Series(False, index=df.index)
        self.pid = Phase(pid_mask, max_step, duration_seconds)
        self.nn = Phase(~pid_mask, max_step, duration_seconds)
        self.eval = Phase(eval_mask, max_step, duration_seconds)


# --- plant ---

class Plant:
    """Физические сигналы установки (трасса железа из exchange-событий)."""

    def __init__(self, df: pd.DataFrame, *, duration_seconds: float, setpoint: float):
        self._df = df
        self._duration = duration_seconds
        self._setpoint = setpoint

    @cached_property
    def time(self) -> Time:
        return Time(self._df.index, self._duration)

    @property
    def process_variable(self) -> pd.Series:
        return self._df["process_variable"]

    @property
    def control_output(self) -> pd.Series:
        return self._df["control_output"]

    @cached_property
    def setpoint(self) -> pd.Series:
        return pd.Series(self._setpoint, index=self._df.index, name="setpoint")

    @cached_property
    def error(self) -> pd.Series:
        return self.setpoint - self.process_variable


# --- interaction: env-состояние, наблюдения, действия, награда, политика ---

class EnvState:
    """Состояние среды по шагам (из env_info), отличное от трассы железа plant."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def process_variable(self) -> pd.Series:
        return self._df["process_variable"]

    @property
    def control_output(self) -> pd.Series:
        return self._df["control_output"]


class Observations:
    """Наблюдения: позиционно ``[i]`` + именованные каналы, каждый в двух формах.

    Состав вектора определяется флагами ``observe_*`` (лог пишет все каналы,
    но в наблюдение агента входят только отмеченные). ``error`` — всегда.
    """

    # (имя, колонка, вид нормировки, флаг-конфига или None для «всегда»)
    _CHANNELS = [
        ("error", "error", "error", None),
        ("prev_error", "prev_error", "error", "observe_prev_error"),
        ("prev_prev_error", "prev_prev_error", "error", "observe_prev_prev_error"),
        ("prev_control_output", "prev_control_output", "control", "observe_prev_control_output"),
        ("prev_prev_control_output", "prev_prev_control_output", "control", "observe_prev_prev_control_output"),
    ]

    def __init__(self, df: pd.DataFrame, *, error_factor: float, control_range: float,
                 observe: dict):
        self._df = df
        self._error_factor = error_factor
        self._control_range = control_range
        self._present = [
            (n, c, k) for (n, c, k, flag) in self._CHANNELS
            if flag is None or observe.get(flag, False)
        ]

    @property
    def names(self) -> list[str]:
        return [n for (n, _, _) in self._present]

    def __len__(self) -> int:
        return len(self._present)

    def __getitem__(self, i: int) -> pd.Series:
        _, col, _ = self._present[i]
        return self._df[col]

    def _factor(self, kind: str) -> float:
        return self._error_factor if kind == "error" else self._control_range

    def _physical(self, name: str) -> pd.Series:
        for n, col, _ in self._present:
            if n == name:
                return self._df[col]
        raise AttributeError(f"В этом прогоне нет наблюдения '{name}'")

    def _norm(self, name: str) -> pd.Series:
        for n, col, kind in self._present:
            if n == name:
                return self._df[col] / self._factor(kind)
        raise AttributeError(f"В этом прогоне нет наблюдения '{name}'")

    @property
    def error(self) -> pd.Series: return self._physical("error")
    @property
    def error_norm(self) -> pd.Series: return self._norm("error")
    @property
    def prev_error(self) -> pd.Series: return self._physical("prev_error")
    @property
    def prev_error_norm(self) -> pd.Series: return self._norm("prev_error")
    @property
    def prev_prev_error(self) -> pd.Series: return self._physical("prev_prev_error")
    @property
    def prev_prev_error_norm(self) -> pd.Series: return self._norm("prev_prev_error")
    @property
    def prev_control_output(self) -> pd.Series: return self._physical("prev_control_output")
    @property
    def prev_control_output_norm(self) -> pd.Series: return self._norm("prev_control_output")


class Actions:
    """Применённое действие: нормированное и физическое (× max_delta)."""

    def __init__(self, df: pd.DataFrame, *, max_delta: float):
        self._df = df
        self._max_delta = max_delta

    @property
    def names(self) -> list[str]:
        return ["control_output_delta"]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, i: int) -> pd.Series:
        if i != 0:
            raise IndexError(i)
        return self.control_output_delta_norm

    @property
    def control_output_delta_norm(self) -> pd.Series:
        return self._df["action_norm"]

    @cached_property
    def control_output_delta(self) -> pd.Series:
        return self._df["action_norm"] * self._max_delta


class Reward:
    """Награда, разложенная на составляющие (как в env_info)."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def _col(self, name: str) -> pd.Series:
        return self._df[name]

    @property
    def total(self) -> pd.Series: return self._col("reward")
    @property
    def error_term(self) -> pd.Series: return self._col("reward_error_term")
    @property
    def action_cost(self) -> pd.Series: return self._col("cost_action")
    @property
    def control_cost(self) -> pd.Series: return self._col("cost_control")
    @property
    def barrier_cost(self) -> pd.Series: return self._col("cost_barrier")
    @property
    def terminal_cost(self) -> pd.Series: return self._col("cost_terminal")
    @property
    def alive(self) -> pd.Series: return self._col("reward_alive")


class Policy:
    """Внутренности политики (только нормированное пространство решений)."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def mean(self) -> pd.Series: return self._df["policy_mean"]
    @property
    def std(self) -> pd.Series: return self._df["policy_std"]
    @property
    def log_prob(self) -> pd.Series: return self._df["policy_log_prob"]
    @property
    def raw(self) -> pd.Series: return self._df["policy_raw"]
    @property
    def mode(self) -> pd.Series: return self._df["policy_mode"]
    @cached_property
    def is_eval(self) -> pd.Series: return self._df["policy_mode"].eq("eval")


class Interaction:
    """Процесс взаимодействия агента со средой (траектория + состояние + фазы)."""

    def __init__(self, df: pd.DataFrame, *, duration_seconds: float,
                 error_factor: float, max_delta: float, control_range: float,
                 pid_steps: int, observe: dict):
        self._df = df
        self._duration = duration_seconds
        self._error_factor = error_factor
        self._max_delta = max_delta
        self._control_range = control_range
        self._pid_steps = pid_steps
        self._observe = observe

    @cached_property
    def env(self) -> EnvState:
        return EnvState(self._df)

    @cached_property
    def observations(self) -> Observations:
        return Observations(self._df, error_factor=self._error_factor,
                            control_range=self._control_range, observe=self._observe)

    @cached_property
    def actions(self) -> Actions:
        return Actions(self._df, max_delta=self._max_delta)

    @cached_property
    def reward(self) -> Reward:
        return Reward(self._df)

    @cached_property
    def policy(self) -> Policy:
        return Policy(self._df)

    @cached_property
    def phases(self) -> Phases:
        return Phases(self._df, self._pid_steps, self._duration)

    @cached_property
    def time(self) -> Time:
        return Time(self._df.index, self._duration)

    @cached_property
    def step_interval(self) -> StepInterval:
        return StepInterval(self._df)

    @property
    def global_step(self) -> pd.Series:
        return self._df.index.to_series().rename("global_step")

    @property
    def episode_step(self) -> pd.Series:
        return self._df["episode_step"]

    @property
    def terminated(self) -> pd.Series:
        return self._df["terminated"]

    @property
    def truncated(self) -> pd.Series:
        return self._df["truncated"]


# --- train ---

class Evaluation:
    """Метрики периодической оценки во время обучения (event=evaluation)."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def __len__(self) -> int:
        return len(self._df)

    @property
    def step(self) -> pd.Series: return self._df["step"]
    @property
    def episodes(self) -> pd.Series: return self._df["episodes"]
    @property
    def reward_mean(self) -> pd.Series: return self._df["reward_mean"]
    @property
    def reward_sum(self) -> pd.Series: return self._df["reward_sum"]
    @property
    def reward_max(self) -> pd.Series: return self._df["reward_max"]
    @property
    def reward_min(self) -> pd.Series: return self._df["reward_min"]


class Train:
    """Метрики процесса обучения (ось train_step)."""

    def __init__(self, df: pd.DataFrame, evaluation_df: pd.DataFrame, *,
                 duration_seconds: float, target_entropy: float | None = None):
        self._df = df
        self._eval_df = evaluation_df
        self._duration = duration_seconds
        self._target_entropy = target_entropy

    @cached_property
    def evaluation(self) -> Evaluation:
        return Evaluation(self._eval_df)

    @cached_property
    def time(self) -> Time:
        return Time(self._df.index, self._duration)

    @property
    def train_step(self) -> pd.Series: return self._df["train_step"]
    @property
    def actor_loss(self) -> pd.Series: return self._df["actor_loss"]
    @property
    def q1_loss(self) -> pd.Series: return self._df["q1_loss"]
    @property
    def q2_loss(self) -> pd.Series: return self._df["q2_loss"]
    @property
    def alpha(self) -> pd.Series: return self._df["alpha"]
    @property
    def alpha_loss(self) -> pd.Series: return self._df["alpha_loss"]
    @property
    def buffer_size(self) -> pd.Series: return self._df["buffer_size"]

    @cached_property
    def entropy(self) -> pd.Series:
        """Энтропия политики: логированная колонка либо восстановление по формуле.

        Старые поколения не писали entropy — она выводится из температуры:
        ``H = alpha_loss / log(alpha) + target_entropy`` (там, где alpha > 0).
        """
        if "entropy" in self._df.columns:
            return self._df["entropy"]
        if self._target_entropy is None:
            raise AttributeError(
                "entropy не логировалась, и нет target_entropy для восстановления"
            )
        alpha = self._df["alpha"]
        log_alpha = np.log(alpha.where(alpha > 0))
        return self._df["alpha_loss"] / log_alpha + self._target_entropy


class Checkpoints:
    """Сохранённые артефакты обучения: агент и буфер воспроизведения.

    Реконструкция агента (через ``build_agent``) и загрузка буфера требуют
    полного конфига и путей — всё это инкапсулировано здесь; наружу торчат
    лишь готовые объекты. Тяжёлые зависимости (torch, rl.*) импортируются
    лениво, чтобы пакет анализа оставался лёгким.
    """

    def __init__(self, dir: Path, raw_config):
        self._dir = Path(dir)
        self._config = raw_config

    @cached_property
    def agent(self):
        from nn_laser_stabilizer.rl.envs.factory import get_spaces_from_config
        from nn_laser_stabilizer.rl.algorithms.factory import build_agent

        agent_dir = self._dir / "agent_sac"
        if not agent_dir.exists():
            raise FileNotFoundError(f"нет сохранённого агента: {agent_dir}")

        obs_space, act_space = get_spaces_from_config(
            self._config.env, seed=int(self._config.seed)
        )
        agent = build_agent(
            algorithm_config=self._config.algorithm,
            observation_space=obs_space,
            action_space=act_space,
        )
        agent.load(agent_dir)
        return agent

    @cached_property
    def replay_buffer(self):
        from nn_laser_stabilizer.rl.data.replay_buffer import ReplayBuffer

        path = self._dir / "data" / "replay_buffer.pth"
        if not path.exists():
            raise FileNotFoundError(f"нет буфера воспроизведения: {path}")
        return ReplayBuffer.load(path)
