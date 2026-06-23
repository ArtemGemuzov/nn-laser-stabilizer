import re
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from nn_laser_stabilizer.analysis.experiment import Experiment
from nn_laser_stabilizer.analysis.sources import read_jsonl
from nn_laser_stabilizer.config.config import load_config


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
    pid: Phase
    nn: Phase
    eval: Phase


@dataclass(frozen=True, eq=False)
class Plant:
    """Физические сигналы установки."""

    time: Time
    process_variable: pd.Series
    control_output: pd.Series
    setpoint: pd.Series
    error: pd.Series


@dataclass(frozen=True, eq=False)
class EnvState:
    """Состояние среды по шагам."""

    process_variable: pd.Series
    control_output: pd.Series


@dataclass(frozen=True, eq=False)
class Observations:
    physical: pd.DataFrame
    normalized: pd.DataFrame

    @property
    def names(self) -> list[str]:
        return list(self.physical.columns)

    def __len__(self) -> int:
        return self.physical.shape[1]

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
    control_output_delta: pd.Series
    control_output_delta_norm: pd.Series

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
    total: pd.Series
    error_term: pd.Series
    action_cost: pd.Series
    control_cost: pd.Series
    barrier_cost: pd.Series
    terminal_cost: pd.Series
    alive: pd.Series


@dataclass(frozen=True, eq=False)
class Policy:
    mean: pd.Series
    std: pd.Series
    log_prob: pd.Series
    raw: pd.Series
    mode: pd.Series

    @property
    def is_eval(self) -> pd.Series:
        return self.mode.eq("eval")


@dataclass(frozen=True, eq=False)
class Interaction:
    time: Time
    step_interval: StepInterval
    env: EnvState
    observations: Observations
    actions: Actions
    reward: Reward
    policy: Policy
    phases: Phases
    global_step: pd.Series
    episode_step: pd.Series
    terminated: pd.Series
    truncated: pd.Series


@dataclass(frozen=True, eq=False)
class Evaluation:
    step: pd.Series
    episodes: pd.Series
    reward_mean: pd.Series
    reward_sum: pd.Series
    reward_max: pd.Series
    reward_min: pd.Series

    def __len__(self) -> int:
        return len(self.step)


@dataclass(frozen=True, eq=False)
class Train:
    time: Time
    evaluation: Evaluation
    train_step: pd.Series
    actor_loss: pd.Series
    q1_loss: pd.Series
    q2_loss: pd.Series
    alpha: pd.Series
    alpha_loss: pd.Series
    buffer_size: pd.Series
    entropy: pd.Series


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


@dataclass(frozen=True, eq=False)
class Params:
    setpoint: float            # физ. единицы (деление /10 спрятано в адаптере)
    error_factor: float        # нормировка error-каналов наблюдений (NaN, если не задана)
    max_delta: float           # масштаб действия (денорм. control_output_delta)
    control_range: float       # control_max − control_min, для денорм. CO-каналов
    pid_steps: int             # граница фазы ПИД-исследования (0, если её нет)
    target_entropy: float      # для восстановления entropy там, где её не логировали
    gamma: float | None        # коэффициент дисконтирования
    observe: dict              # флаги состава вектора наблюдений (observe_*)


# ═══════════════════════════════════════════════════ канонизация (file → table)

# raw-поле (после flatten в read_jsonl) → каноническая колонка.
# Поля политики (списки [x]) помечены в _STEP_SCALAR_COLS для извлечения скаляра.
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
_STEP_SCALAR_COLS = ("policy_mean", "policy_std", "policy_log_prob", "policy_raw")
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

_OBS_CHANNELS = [
    ("error", "error", "error", None),
    ("prev_error", "prev_error", "error", "observe_prev_error"),
    ("prev_prev_error", "prev_prev_error", "error", "observe_prev_prev_error"),
    ("prev_control_output", "prev_control_output", "control", "observe_prev_control_output"),
    ("prev_prev_control_output", "prev_prev_control_output", "control", "observe_prev_prev_control_output"),
]

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


def _canon(
    df: pd.DataFrame, rename: dict[str, str], scalar_cols: tuple[str, ...] = ()
) -> pd.DataFrame:
    present = {raw: canon for raw, canon in rename.items() if raw in df.columns}
    out = df[list(present)].rename(columns=present).reset_index(drop=True)
    for col in scalar_cols:
        if col in out.columns:
            out[col] = out[col].map(_first_scalar)
    # global_step — позиционный счётчик
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


def _seconds(index: pd.Index, duration_seconds: float) -> pd.Series:
    """Ось времени из global_step-индекса (приближённо, по полной длительности)."""
    gs = index.to_series().astype(float)
    return gs / gs.max() * duration_seconds


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index, name=name)


class NeuralControllerExperiment(Experiment):
    @cached_property
    def _raw(self) -> pd.DataFrame:
        # collector.jsonl несёт два потока в одном файле (step + exchange);
        # сырой парс кешируется, поэтому файл не читается дважды.
        return read_jsonl(self._dir / "collector.jsonl")

    @cached_property
    def _step(self) -> pd.DataFrame:
        rows = self._raw[self._raw["event"] == "step"]
        return _canon(rows, _STEP_RENAME, _STEP_SCALAR_COLS)

    @cached_property
    def _exchange(self) -> pd.DataFrame:
        rows = self._raw[self._raw["event"] == "exchange"]
        return _canon(rows, _EXCHANGE_RENAME)

    @cached_property
    def _train_raw(self) -> pd.DataFrame:
        # train.jsonl смешивает event=step (метрики) и event=evaluation (награды)
        return read_jsonl(self._dir / "train.jsonl")

    def _train_subset(self, event: str, rename: dict[str, str]) -> pd.DataFrame:
        raw = self._train_raw
        rows = raw[raw["event"] == event] if "event" in raw.columns else raw
        return _canon(rows, rename)

    @cached_property
    def _train(self) -> pd.DataFrame:
        return self._train_subset("step", _TRAIN_RENAME)

    @cached_property
    def _evaluation(self) -> pd.DataFrame:
        return self._train_subset("evaluation", _EVAL_RENAME)

    @cached_property
    def _config(self):
        return load_config(self._dir / "config.yaml")

    @cached_property
    def _duration(self) -> float:
        return _read_duration_seconds(self._dir / "console.log")

    @cached_property
    def params(self) -> Params:
        c = self._config
        args = c.env.args
        gamma = c.get("algorithm.gamma")
        return Params(
            setpoint=float(args.setpoint) / 10,
            # отсутствует в части прогонов → NaN (норм. error-каналов тогда NaN)
            error_factor=float(args.get("error_normalixation_factor", float("nan"))), # TODO: найти старые параметры
            max_delta=float(args.action.max_delta),
            control_range=float(args.control_max) - float(args.control_min),
            # exploration есть не всегда (например, в инференсе)
            pid_steps=int(c.get("exploration.steps", 0)),
            # старые эксперименты не писали целевую энтропию
            target_entropy=float(c.get("algorithm.target_entropy_value", -1)),
            gamma=float(gamma) if gamma is not None else None,
            observe={
                flag: bool(args.get(flag, False))
                for flag in (
                    "observe_prev_error",
                    "observe_prev_prev_error",
                    "observe_prev_control_output",
                    "observe_prev_prev_control_output",
                )
            },
        )

    @property
    def plant_process_variable(self) -> pd.Series:
        return _col(self._exchange, "process_variable")

    @property
    def plant_control_output(self) -> pd.Series:
        return _col(self._exchange, "control_output")

    @property
    def plant_setpoint(self) -> pd.Series:
        return pd.Series(self.params.setpoint, index=self._exchange.index, name="setpoint")

    @property
    def plant_error(self) -> pd.Series:
        return self.plant_setpoint - self.plant_process_variable

    @property
    def plant_time(self) -> Time:
        return Time(_seconds(self._exchange.index, self._duration))

    @property
    def env_process_variable(self) -> pd.Series:
        return _col(self._step, "process_variable")

    @property
    def env_control_output(self) -> pd.Series:
        return _col(self._step, "control_output")

    @property
    def interaction_time(self) -> Time:
        return Time(_seconds(self._step.index, self._duration))

    @property
    def step_interval(self) -> StepInterval:
        return StepInterval(_col(self._step, "step_interval_us"))

    @property
    def observations(self) -> Observations:
        df = self._step
        observe = self.params.observe
        present = [
            (name, col, kind)
            for (name, col, kind, flag) in _OBS_CHANNELS
            if flag is None or observe.get(flag, False)
        ]
        factor = {"error": self.params.error_factor, "control": self.params.control_range}
        physical = pd.DataFrame({name: _col(df, col) for (name, col, _) in present})
        normalized = pd.DataFrame(
            {name: _col(df, col) / factor[kind] for (name, col, kind) in present}
        )
        return Observations(physical=physical, normalized=normalized)

    @property
    def actions(self) -> Actions:
        action_norm = _col(self._step, "action_norm")
        return Actions(
            control_output_delta=action_norm * self.params.max_delta,
            control_output_delta_norm=action_norm,
        )

    @property
    def reward(self) -> Reward:
        df = self._step
        return Reward(
            total=_col(df, "reward"),
            error_term=_col(df, "reward_error_term"),
            action_cost=_col(df, "cost_action"),
            control_cost=_col(df, "cost_control"),
            barrier_cost=_col(df, "cost_barrier"),
            terminal_cost=_col(df, "cost_terminal"),
            alive=_col(df, "reward_alive"),
        )

    @property
    def policy(self) -> Policy:
        df = self._step
        return Policy(
            mean=_col(df, "policy_mean"),
            std=_col(df, "policy_std"),
            log_prob=_col(df, "policy_log_prob"),
            raw=_col(df, "policy_raw"),
            mode=_col(df, "policy_mode"),
        )

    @property
    def phases(self) -> Phases:
        df = self._step
        gs = df.index.to_series()
        max_step = int(gs.max())
        pid_mask = gs <= self.params.pid_steps
        if "policy_mode" in df.columns:
            eval_mask = df["policy_mode"].eq("eval").fillna(False)
        else:
            eval_mask = pd.Series(False, index=df.index)
        dur = self._duration
        return Phases(
            pid=Phase(pid_mask, max_step, dur),
            nn=Phase(~pid_mask, max_step, dur),
            eval=Phase(eval_mask, max_step, dur),
        )

    @property
    def env(self) -> EnvState:
        return EnvState(
            process_variable=self.env_process_variable,
            control_output=self.env_control_output,
        )

    @property
    def train_entropy(self) -> pd.Series:
        """Энтропия политики: логированная колонка либо восстановление по формуле.

        Старые эксперименты не записывали энтропию, но ее можно вычислить:
        ``H = alpha_loss / log(alpha) + target_entropy`` (там, где alpha > 0).
        """
        df = self._train
        if "entropy" in df.columns:
            return df["entropy"]
        alpha = df["alpha"]
        log_alpha = np.log(alpha.where(alpha > 0))
        return df["alpha_loss"] / log_alpha + self.params.target_entropy

    @property
    def evaluation(self) -> Evaluation:
        df = self._evaluation
        return Evaluation(
            step=_col(df, "step"),
            episodes=_col(df, "episodes"),
            reward_mean=_col(df, "reward_mean"),
            reward_sum=_col(df, "reward_sum"),
            reward_max=_col(df, "reward_max"),
            reward_min=_col(df, "reward_min"),
        )

    @property
    def plant(self) -> Plant:
        return Plant(
            time=self.plant_time,
            process_variable=self.plant_process_variable,
            control_output=self.plant_control_output,
            setpoint=self.plant_setpoint,
            error=self.plant_error,
        )

    @property
    def interaction(self) -> Interaction:
        df = self._step
        return Interaction(
            time=self.interaction_time,
            step_interval=self.step_interval,
            env=self.env,
            observations=self.observations,
            actions=self.actions,
            reward=self.reward,
            policy=self.policy,
            phases=self.phases,
            global_step=df.index.to_series().rename("global_step"),
            episode_step=_col(df, "episode_step"),
            terminated=_col(df, "terminated"),
            truncated=_col(df, "truncated"),
        )

    @property
    def train(self) -> Train:
        df = self._train
        return Train(
            time=Time(_seconds(df.index, self._duration)),
            evaluation=self.evaluation,
            train_step=_col(df, "train_step"),
            actor_loss=_col(df, "actor_loss"),
            q1_loss=_col(df, "q1_loss"),
            q2_loss=_col(df, "q2_loss"),
            alpha=_col(df, "alpha"),
            alpha_loss=_col(df, "alpha_loss"),
            buffer_size=_col(df, "buffer_size"),
            entropy=self.train_entropy,
        )

    @property
    def checkpoints(self) -> Checkpoints:
        return Checkpoints(self._dir, self._config)
