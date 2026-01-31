from functools import partial
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch

from nn_laser_stabilizer.config.config import Config, find_and_load_config
from nn_laser_stabilizer.data.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.envs.env_wrapper import TorchEnvWrapper
from nn_laser_stabilizer.envs.neural_controller_env import NeuralControllerEnv
from nn_laser_stabilizer.envs.plant_backend import MockPlantBackend
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.logger import NoOpLogger
from nn_laser_stabilizer.normalize import normalize_to_minus1_plus1
from nn_laser_stabilizer.paths import WorkingDirectoryContext


def csv_to_buffer_via_env(
    *,
    context: ExperimentContext,
    env_config_path: Path,
    csv_path: Path
) -> None:
    config = find_and_load_config(env_config_path)
    env_args = config.args

    control_min = int(env_args.control_min)
    control_max = int(env_args.control_max)
    normalize_control = partial(
        normalize_to_minus1_plus1,
        min_val=float(control_min),
        max_val=float(control_max),
    )

    process_variable_max = int(env_args.process_variable_max)
    setpoint = int(env_args.setpoint)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    num_rows = len(df)
    if num_rows < 3:
        raise ValueError(
            f"CSV must have at least 3 rows for one transition via env, got {num_rows}"
        )

    process_variables = df["process_variable"].to_numpy(dtype=np.int64)
    control_outputs = df["control_output"].to_numpy(dtype=np.int64)
    n = len(process_variables)
    index = 1

    def reset_fn() -> tuple[int, int, int]:
        nonlocal index
        index = 1
        return (
            int(process_variables[1]),
            setpoint,
            int(control_outputs[0]),
        )

    def exchange_fn(control_output: int) -> int:
        nonlocal index
        if index >= n - 1:
            return int(process_variables[n - 1])
        result = int(process_variables[index + 1])
        index += 1
        return result

    backend = MockPlantBackend(
        reset_fn=reset_fn,
        exchange_fn=exchange_fn,
        setpoint=setpoint,
    )
    base_logger = NoOpLogger()
    base_env = NeuralControllerEnv(
        backend=backend,
        base_logger=base_logger,
        control_min=control_min,
        control_max=control_max,
        process_variable_max=process_variable_max,
    )
    env = TorchEnvWrapper(base_env)

    obs_dim = env.observation_space.dim
    action_dim = env.action_space.dim
    num_steps = num_rows - 2
    buffer = ReplayBuffer(
        capacity=num_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    obs, _ = env.reset()
    for step_idx in range(num_steps):
        control_output = int(control_outputs[step_idx + 1])
        action_norm = normalize_control(float(control_output))
        action = torch.tensor([action_norm], dtype=torch.float32)

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

    output_path = Path("replay_buffer.pth")
    buffer.save(output_path)
    context.logger.log(
        f"Buffer saved: {output_path} (size={len(buffer)}, capacity={buffer.capacity})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CSV to replay buffer via NeuralControllerEnv.",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        required=True,
        help="Path to environment config (e.g. envs/neural_controller or neural_controller).",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Path to CSV file (process_variable, control_output columns).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output .pth buffer file. Default: <experiment_dir>/replay_buffer.pth",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiment_config = Config({
        "experiment_name": "csv_to_buffer_via_env",
        "env_config": args.env_config,
        "csv_path": args.csv_path,
    })
    with ExperimentContext(experiment_config) as context, WorkingDirectoryContext(context.experiment_dir):
        context.logger.log(
            f"csv_to_buffer_via_env: env_config={args.env_config} csv={args.csv_path}"
        )
        csv_to_buffer_via_env(
            context=context,
            env_config_path=args.env_config,
            csv_path=args.csv_path,
        )
        context.logger.log("csv_to_buffer_via_env: done.")
