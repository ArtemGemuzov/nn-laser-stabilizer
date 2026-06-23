import re
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from nn_laser_stabilizer.analysis.experiment import Experiment
from nn_laser_stabilizer.analysis.sources import read_jsonl
from nn_laser_stabilizer.config.config import Config, load_config


@dataclass(frozen=True, eq=False)
class Params:
    config: Config

    @property
    def _args(self):
        return self.config.env.args

    @property
    def setpoint(self) -> float:
        return float(self._args.setpoint) / 10   # /10 — историческая нормировка лога

    @property
    def error_factor(self) -> float:
        return float(self._args.error_normalixation_factor)

    @property
    def max_delta(self) -> float:
        return float(self._args.action.max_delta)

    @property
    def control_range(self) -> float:
        return float(self._args.control_max) - float(self._args.control_min)

    @property
    def pid_steps(self) -> int:
        # exploration есть не всегда (например, в инференсе) → по умолчанию 0
        return int(self.config.get("exploration.steps", 0))

    @property
    def target_entropy(self) -> float:
        return float(self.config.get("algorithm.target_entropy_value", -1))

    @property
    def gamma(self) -> float | None:
        return float(self.config.algorithm.gamma)
    
    @property
    def observe(self) -> dict:
        return {
            flag: bool(self._args.get(flag, False))
            for flag in (
                "observe_prev_error",
                "observe_prev_prev_error",
                "observe_prev_control_output",
                "observe_prev_prev_control_output",
            )
        }


@dataclass(frozen=True, eq=False)
class Time:
    """Время оси (приближённый пересчёт global_step → длительность), с единицами."""

    seconds: pd.Series

    @property
    def minutes(self) -> pd.Series:
        return self.seconds / 60.0

    @property
    def hours(self) -> pd.Series:
        return self.seconds / 3600.0


@dataclass(frozen=True, eq=False)
class TimePoint:
    steps: int
    max_step: int
    duration_seconds: float

    @property
    def seconds(self) -> float:
        if not self.max_step:
            return float("nan")
        return self.steps / self.max_step * self.duration_seconds

    @property
    def minutes(self) -> float:
        return self.seconds / 60.0

    @property
    def hours(self) -> float:
        return self.seconds / 3600.0


@dataclass(frozen=True, eq=False)
class StepInterval:
    microseconds: pd.Series

    @property
    def milliseconds(self) -> pd.Series:
        return self.microseconds / 1e3

    @property
    def seconds(self) -> pd.Series:
        return self.microseconds / 1e6


@dataclass(frozen=True, eq=False)
class Phase:
    mask: pd.Series
    max_step: int
    duration_seconds: float

    @property
    def ranges(self) -> list[tuple[int, int]]:
        return _contiguous_ranges(self.mask)

    @property
    def start(self) -> TimePoint | None:
        r = self.ranges
        return TimePoint(r[0][0], self.max_step, self.duration_seconds) if r else None

    @property
    def end(self) -> TimePoint | None:
        r = self.ranges
        return TimePoint(r[-1][1], self.max_step, self.duration_seconds) if r else None


@dataclass(frozen=True, eq=False)
class Phases:
    step: pd.DataFrame
    params: Params
    duration_seconds: float

    @property
    def _gs(self) -> pd.Series:
        return self.step.index.to_series()

    @property
    def _max_step(self) -> int:
        return int(self._gs.max())

    @property
    def _pid_mask(self) -> pd.Series:
        return self._gs <= self.params.pid_steps

    @property
    def pid(self) -> Phase:
        return Phase(self._pid_mask, self._max_step, self.duration_seconds)

    @property
    def nn(self) -> Phase:
        return Phase(~self._pid_mask, self._max_step, self.duration_seconds)

    @property
    def eval(self) -> Phase:
        if "policy_policy_mode" in self.step.columns:
            mask = self.step["policy_policy_mode"].eq("eval").fillna(False)
        else:
            mask = pd.Series(False, index=self.step.index)
        return Phase(mask, self._max_step, self.duration_seconds)


@dataclass(frozen=True, eq=False)
class Plant:
    exchange: pd.DataFrame
    params: Params
    duration_seconds: float

    @property
    def time(self) -> Time:
        return Time(_seconds(self.exchange.index, self.duration_seconds))

    @property
    def process_variable(self) -> pd.Series:
        return _col(self.exchange, "process_variable")

    @property
    def control_output(self) -> pd.Series:
        return _col(self.exchange, "control_output")

    @property
    def setpoint(self) -> pd.Series:
        return pd.Series(self.params.setpoint, index=self.exchange.index, name="setpoint")

    @property
    def error(self) -> pd.Series:
        return self.setpoint - self.process_variable


@dataclass(frozen=True, eq=False)
class EnvState:
    """Состояние среды по шагам."""

    step: pd.DataFrame

    @property
    def process_variable(self) -> pd.Series:
        return _col(self.step, "process_variable")

    @property
    def control_output(self) -> pd.Series:
        return _col(self.step, "control_output")


@dataclass(frozen=True, eq=False)
class Observations:
    step: pd.DataFrame
    params: Params

    _CHANNELS = [
        ("error", "error", "error", None),
        ("prev_error", "prev_error", "error", "observe_prev_error"),
        ("prev_prev_error", "prev_prev_error", "error", "observe_prev_prev_error"),
        ("prev_control_output", "prev_control_output", "control", "observe_prev_control_output"),
        ("prev_prev_control_output", "prev_prev_control_output", "control", "observe_prev_prev_control_output"),
    ]

    @property
    def _present(self) -> list[tuple[str, str, str]]:
        observe = self.params.observe
        return [
            (name, col, kind)
            for (name, col, kind, flag) in self._CHANNELS
            if flag is None or observe.get(flag, False)
        ]

    @property
    def physical(self) -> pd.DataFrame:
        return pd.DataFrame({name: _col(self.step, col) for (name, col, _) in self._present})

    @property
    def normalized(self) -> pd.DataFrame:
        factor = {"error": self.params.error_factor, "control": self.params.control_range}
        return pd.DataFrame(
            {name: _col(self.step, col) / factor[kind] for (name, col, kind) in self._present}
        )

    @property
    def names(self) -> list[str]:
        return [name for (name, _, _) in self._present]

    def __len__(self) -> int:
        return len(self._present)

    def __getitem__(self, i: int) -> pd.Series:
        return self.physical.iloc[:, i]

    @property
    def error(self) -> pd.Series: return self.physical["error"]
    @property
    def error_norm(self) -> pd.Series: return self.normalized["error"]
    @property
    def prev_error(self) -> pd.Series: return self.physical["prev_error"]
    @property
    def prev_error_norm(self) -> pd.Series: return self.normalized["prev_error"]
    @property
    def prev_prev_error(self) -> pd.Series: return self.physical["prev_prev_error"]
    @property
    def prev_prev_error_norm(self) -> pd.Series: return self.normalized["prev_prev_error"]
    @property
    def prev_control_output(self) -> pd.Series: return self.physical["prev_control_output"]
    @property
    def prev_control_output_norm(self) -> pd.Series: return self.normalized["prev_control_output"]
    @property
    def prev_prev_control_output(self) -> pd.Series: return self.physical["prev_prev_control_output"]
    @property
    def prev_prev_control_output_norm(self) -> pd.Series: return self.normalized["prev_prev_control_output"]


@dataclass(frozen=True, eq=False)
class Actions:
    step: pd.DataFrame
    params: Params

    @property
    def control_output_delta_norm(self) -> pd.Series:
        return _col(self.step, "action_norm")

    @property
    def control_output_delta(self) -> pd.Series:
        return self.control_output_delta_norm * self.params.max_delta

    @property
    def names(self) -> list[str]:
        return ["control_output_delta"]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, i: int) -> pd.Series:
        if i != 0:
            raise IndexError(i)
        return self.control_output_delta_norm


@dataclass(frozen=True, eq=False)
class Reward:
    step: pd.DataFrame

    @property
    def total(self) -> pd.Series: return _col(self.step, "reward")
    @property
    def error_term(self) -> pd.Series: return _col(self.step, "reward_error_term")
    @property
    def action_cost(self) -> pd.Series: return _col(self.step, "cost_action")
    @property
    def control_cost(self) -> pd.Series: return _col(self.step, "cost_control")
    @property
    def barrier_cost(self) -> pd.Series: return _col(self.step, "cost_barrier")
    @property
    def terminal_cost(self) -> pd.Series: return _col(self.step, "cost_terminal")
    @property
    def alive(self) -> pd.Series: return _col(self.step, "reward_alive")


@dataclass(frozen=True, eq=False)
class Policy:
    step: pd.DataFrame

    @property
    def mean(self) -> pd.Series: return _scalar_col(self.step, "policy_mean_action")
    @property
    def std(self) -> pd.Series: return _scalar_col(self.step, "policy_std")
    @property
    def log_prob(self) -> pd.Series: return _scalar_col(self.step, "policy_log_prob")
    @property
    def raw(self) -> pd.Series: return _scalar_col(self.step, "policy_raw_action")
    @property
    def mode(self) -> pd.Series: return _col(self.step, "policy_policy_mode")
    @property
    def is_eval(self) -> pd.Series: return self.mode.eq("eval")


@dataclass(frozen=True, eq=False)
class Interaction:
    step: pd.DataFrame
    params: Params
    duration_seconds: float

    @property
    def time(self) -> Time:
        return Time(_seconds(self.step.index, self.duration_seconds))

    @property
    def step_interval(self) -> StepInterval:
        return StepInterval(_col(self.step, "step_interval_us"))

    @property
    def env(self) -> EnvState:
        return EnvState(self.step)

    @property
    def observations(self) -> Observations:
        return Observations(self.step, self.params)

    @property
    def actions(self) -> Actions:
        return Actions(self.step, self.params)

    @property
    def reward(self) -> Reward:
        return Reward(self.step)

    @property
    def policy(self) -> Policy:
        return Policy(self.step)

    @property
    def phases(self) -> Phases:
        return Phases(self.step, self.params, self.duration_seconds)

    @property
    def global_step(self) -> pd.Series:
        return self.step.index.to_series().rename("global_step")

    @property
    def episode_step(self) -> pd.Series:
        return _col(self.step, "step")  # сырое имя в env_info — "step"

    @property
    def terminated(self) -> pd.Series:
        return _col(self.step, "terminated")

    @property
    def truncated(self) -> pd.Series:
        return _col(self.step, "truncated")


@dataclass(frozen=True, eq=False)
class Evaluation:
    frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def step(self) -> pd.Series: return _col(self.frame, "step")
    @property
    def episodes(self) -> pd.Series: return _col(self.frame, "episodes")
    @property
    def reward_mean(self) -> pd.Series: return _col(self.frame, "reward_mean")
    @property
    def reward_sum(self) -> pd.Series: return _col(self.frame, "reward_sum")
    @property
    def reward_max(self) -> pd.Series: return _col(self.frame, "reward_max")
    @property
    def reward_min(self) -> pd.Series: return _col(self.frame, "reward_min")


@dataclass(frozen=True, eq=False)
class Train:
    frame: pd.DataFrame          
    eval_frame: pd.DataFrame
    params: Params
    duration_seconds: float

    @property
    def time(self) -> Time:
        return Time(_seconds(self.frame.index, self.duration_seconds))

    @property
    def evaluation(self) -> Evaluation:
        return Evaluation(self.eval_frame)

    @property
    def train_step(self) -> pd.Series: return _col(self.frame, "step")
    @property
    def actor_loss(self) -> pd.Series: return _col(self.frame, "actor_loss")
    @property
    def q1_loss(self) -> pd.Series: return _col(self.frame, "loss_q1")
    @property
    def q2_loss(self) -> pd.Series: return _col(self.frame, "loss_q2")
    @property
    def alpha(self) -> pd.Series: return _col(self.frame, "alpha")
    @property
    def alpha_loss(self) -> pd.Series: return _col(self.frame, "alpha_loss")
    @property
    def buffer_size(self) -> pd.Series: return _col(self.frame, "buffer_size")

    @property
    def entropy(self) -> pd.Series:
        """Энтропия политики: логированная колонка либо восстановление по формуле.

        ``H = alpha_loss / log(alpha) + target_entropy`` (там, где alpha > 0).
        """
        df = self.frame
        if "entropy" in df.columns:
            return df["entropy"]
        alpha = df["alpha"]
        log_alpha = np.log(alpha.where(alpha > 0))
        return df["alpha_loss"] / log_alpha + self.params.target_entropy


class Checkpoints:
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


# Таймстемп в начале строки console.log: "[2026-06-09 17:49:47,123] ..."
_TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:,\d{3})?\]")


def _first_scalar(value) -> float:
    """Достать скаляр из значения-списка ``[x]`` (поля политики логируются так)."""
    if isinstance(value, (list, tuple)) and value:
        return float(value[0])
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


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


def _with_global_step(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True)
    out.index = pd.RangeIndex(1, len(out) + 1, name="global_step")
    return out


def _contiguous_ranges(mask: pd.Series) -> list[tuple[int, int]]:
    """Непрерывные диапазоны (start_gs, end_gs) по индексу там, где маска True."""
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    prev: int | None = None
    for gs, flag in zip(mask.index, mask.to_numpy()):
        if flag and start is None:
            start = gs
        if not flag and start is not None:
            if prev is None:
                raise RuntimeError(
                    "закрытие диапазона при prev=None — нарушен инвариант обхода"
                )
            ranges.append((start, prev))
            start = None
        prev = gs
    if start is not None:
        if prev is None:
            raise RuntimeError(
                "открытый диапазон при prev=None — нарушен инвариант обхода"
            )
        ranges.append((start, prev))
    return ranges


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index, name=name)


def _scalar_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Колонка-вектор ``[x]`` (поля политики логируются списками) → скаляр в каждой ячейке."""
    return _col(df, name).map(_first_scalar)


def _seconds(index: pd.Index, duration_seconds: float) -> pd.Series:
    """Ось времени из global_step-индекса (приближённо, по полной длительности)."""
    gs = index.to_series().astype(float)
    return gs / gs.max() * duration_seconds


@dataclass(frozen=True, eq=False)
class _CollectorTables:
    """Канонические таблицы из collector.jsonl (один файл → два потока событий)."""

    step: pd.DataFrame
    exchange: pd.DataFrame


@dataclass(frozen=True, eq=False)
class _TrainTables:
    """Канонические таблицы из train.jsonl (метрики обучения + оценки)."""

    train: pd.DataFrame
    evaluation: pd.DataFrame


# ═══════════════════════════════════════════════════════════════ эксперимент

class NeuralControllerExperiment(Experiment):
    @cached_property
    def _collector(self) -> _CollectorTables:
        raw = read_jsonl(self._dir / "collector.jsonl")
        return _CollectorTables(
            step=_with_global_step(raw[raw["event"] == "step"]),
            exchange=_with_global_step(raw[raw["event"] == "exchange"]),
        )

    @cached_property
    def _training(self) -> _TrainTables:
        raw = read_jsonl(self._dir / "train.jsonl")

        def rows(event: str) -> pd.DataFrame:
            return raw[raw["event"] == event] if "event" in raw.columns else raw

        return _TrainTables(
            train=_with_global_step(rows("step")),
            evaluation=_with_global_step(rows("evaluation")),
        )

    @cached_property
    def _config(self) -> Config:
        return load_config(self._dir / "config.yaml")

    @cached_property
    def _duration(self) -> float:
        return _read_duration_seconds(self._dir / "console.log")

    @property
    def params(self) -> Params:
        return Params(self._config)

    @property
    def plant(self) -> Plant:
        return Plant(self._collector.exchange, self.params, self._duration)

    @property
    def interaction(self) -> Interaction:
        return Interaction(self._collector.step, self.params, self._duration)

    @property
    def train(self) -> Train:
        t = self._training
        return Train(t.train, t.evaluation, self.params, self._duration)

    @property
    def checkpoints(self) -> Checkpoints:
        return Checkpoints(self._dir, self._config)
