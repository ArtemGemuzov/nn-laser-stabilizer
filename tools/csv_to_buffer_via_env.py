from functools import partial
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch

from nn_laser_stabilizer.config.config import find_and_load_config
from nn_laser_stabilizer.paths import find_project_root
from nn_laser_stabilizer.experiment.decorator import experiment
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.logger import NoOpLogger
from nn_laser_stabilizer.normalize import normalize_to_minus1_plus1
from nn_laser_stabilizer.rl.data.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.rl.envs.env_wrapper import TorchEnvWrapper
from nn_laser_stabilizer.rl.envs.neural_controller import ActionType, NeuralController
from nn_laser_stabilizer.rl.envs.plant_backend import MockPlantBackend

BUFFER_RESET_STEPS = 3


def _load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".jsonl", ".json"):
        return pd.read_json(path, lines=True)
    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            f"Expected .csv, .jsonl, or .json"
        )


def _make_extra_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert data log to replay buffer via NeuralController with MockPlantBackend.",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        required=True,
        help="Path to environment config (e.g. envs/neural_controller).",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data file (.csv or .jsonl) with process_variable and control_output columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="replay_buffer.pth",
        help="Path to output .pth buffer file. Default: replay_buffer.pth",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        default=False,
        help="Filter out rows where is_warming_up is true.",
    )
    return parser


@experiment(
    experiment_name="csv_to_buffer_via_env",
    extra_parser=_make_extra_parser()
)
def main(context: ExperimentContext) -> None:
    cli = context.config.cli
    env_config_path = Path(cli.env_config)
    data_path = Path(cli.data)
    if not data_path.is_absolute():
        data_path = (find_project_root() / data_path).resolve()

    env_config = find_and_load_config(env_config_path)
    env_args = env_config.args

    control_min = int(env_args.control_min)
    control_max = int(env_args.control_max)
    process_variable_max = int(env_args.process_variable_max)
    setpoint = int(env_args.setpoint)

    action_type = ActionType(str(env_args.action.type))
    if action_type != ActionType.DELTA:
        raise ValueError(
            f"csv_to_buffer_via_env supports only action type "
            f"'{ActionType.DELTA.value}', got '{action_type.value}'"
        )
    max_action_delta = int(env_args.action.max_delta)
    normalize_delta = partial(
        normalize_to_minus1_plus1,
        min_val=-float(max_action_delta),
        max_val=float(max_action_delta),
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = _load_dataframe(data_path)
    context.logger.log(f"Loaded {len(df)} rows from {data_path}")

    if cli.skip_warmup and "is_warming_up" in df.columns:
        warmup_count = df["is_warming_up"].sum()
        df = df[~df["is_warming_up"]].reset_index(drop=True)
        context.logger.log(f"Filtered out {warmup_count} warmup rows, {len(df)} rows remaining")

    num_rows = len(df)
    if num_rows < 3:
        raise ValueError(
            f"Data must have at least 3 rows for one transition via env, got {num_rows}"
        )

    process_variables = df["process_variable"].to_numpy(dtype=np.int64)
    control_outputs = df["control_output"].to_numpy(dtype=np.int64)
    n = len(process_variables)
    index = 0

    def exchange_fn(control_output: int) -> int:
        nonlocal index
        if index >= n - 1:
            return int(process_variables[n - 1])
        result = int(process_variables[index + 1])
        index += 1
        return result

    backend = MockPlantBackend(
        exchange_fn=exchange_fn,
        setpoint=setpoint,
    )
    base_logger = NoOpLogger()
    base_env = NeuralController(
        backend=backend,
        base_logger=base_logger,
        control_min=control_min,
        control_max=control_max,
        process_variable_max=process_variable_max,
        reset_value=int(control_outputs[0]),
        reset_steps=BUFFER_RESET_STEPS,
        observe_prev_error=bool(env_args.get("observe_prev_error", True)),
        observe_prev_prev_error=bool(env_args.get("observe_prev_prev_error", True)),
        observe_control_output=bool(env_args.get("observe_control_output", True)),
        action_type=action_type,
        max_action_delta=max_action_delta,
    )
    env = TorchEnvWrapper(base_env)

    obs_dim = env.observation_space.dim
    action_dim = env.action_space.dim
    offset = BUFFER_RESET_STEPS - 1
    num_steps = num_rows - BUFFER_RESET_STEPS - 1
    buffer = ReplayBuffer(
        capacity=num_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    obs, _ = env.reset()
    for step_idx in range(num_steps):
        data_idx = offset + step_idx
        control_prev = int(control_outputs[data_idx])
        control_curr = int(control_outputs[data_idx + 1])
        delta = control_curr - control_prev
        delta = max(-max_action_delta, min(max_action_delta, delta))
        delta_norm = normalize_delta(float(delta))
        action = torch.tensor([delta_norm], dtype=torch.float32)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.add(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            done=done,
        )
        obs = next_obs

    env.close()

    output_path = Path(cli.output)
    buffer.save(output_path)
    context.logger.log(
        f"Buffer saved: {output_path} (size={len(buffer)}, capacity={buffer.capacity})"
    )


if __name__ == "__main__":
    main()
