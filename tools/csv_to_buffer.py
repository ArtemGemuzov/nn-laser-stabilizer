import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from nn_laser_stabilizer.experiment.decorator import experiment
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.rl.data.buffer_loader import load_buffer_from_csv


def make_extract_transition(
    process_variable_max: int,
    control_min: int,
    control_max: int,
    setpoint: int,
):
    span = float(control_max - control_min)
    setpoint_norm = setpoint / process_variable_max

    def _normalize_control_output(control_output: int) -> float:
        norm_01 = float(control_output - control_min) / span
        return 2.0 * norm_01 - 1.0

    def extract_transition(
        df: pd.DataFrame,
        idx: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        if idx == 0 or idx >= len(df) - 1:
            return None

        current_row = df.iloc[idx]
        next_row = df.iloc[idx + 1]
        prev_row = df.iloc[idx - 1]

        process_variable = float(current_row["process_variable"])
        process_variable_norm = np.clip(process_variable / process_variable_max, 0.0, 1.0)
        error = setpoint_norm - process_variable_norm

        control_output_prev = int(prev_row["control_output"])
        control_output_prev_norm = _normalize_control_output(control_output_prev)

        observation = np.array([error, control_output_prev_norm], dtype=np.float32)

        control_output_t = int(current_row["control_output"])
        control_output_t_norm = _normalize_control_output(control_output_t)

        action = np.array([control_output_t_norm], dtype=np.float32)

        next_process_variable = float(next_row["process_variable"])
        next_process_variable_norm = np.clip(
            next_process_variable / process_variable_max, 0.0, 1.0
        )
        next_error = setpoint_norm - next_process_variable_norm

        next_observation = np.array([next_error, control_output_t_norm], dtype=np.float32)

        reward = 1.0 - 2.0 * abs(next_error)

        done = False

        return observation, action, float(reward), next_observation, done

    return extract_transition


def _make_extra_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert CSV to replay buffer (no row trimming).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file. Overrides config csv_data.csv_path if set.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for .pth buffer file. Overrides config csv_data.buffer_output if set.",
    )
    return parser


@experiment(
    experiment_name="csv_to_buffer", 
    config_name="csv_to_buffer", 
    extra_parser=_make_extra_parser()
)
def main(context: ExperimentContext) -> None:
    config = context.config
    base_config = config.base_config
    csv_data = config.csv_data

    cli = context.config.cli
    csv_path = cli.get("csv")
    csv_path = Path(csv_path) if csv_path is not None else Path(csv_data.csv_path)
    csv_path = csv_path.resolve()

    output_path = cli.get("output")
    output_path = (
        Path(output_path)
        if output_path is not None
        else Path(csv_data.get("buffer_output", "data/replay_buffer.pth"))
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    process_variable_max = int(base_config.env.args.process_variable_max)
    control_min = int(base_config.env.args.control_min)
    control_max = int(base_config.env.args.control_max)
    setpoint = int(base_config.env.args.setpoint)

    extract_transition_fn = make_extract_transition(
        process_variable_max=process_variable_max,
        control_min=control_min,
        control_max=control_max,
        setpoint=setpoint,
    )
    buffer = load_buffer_from_csv(
        csv_path=csv_path,
        extract_transition=extract_transition_fn,
    )
    buffer.save(output_path)
    context.logger.log(
        f"Buffer saved: {output_path} (size={len(buffer)}, capacity={buffer.capacity})"
    )


if __name__ == "__main__":
    main()
