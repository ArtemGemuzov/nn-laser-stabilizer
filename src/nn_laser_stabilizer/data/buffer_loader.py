from typing import Callable, Optional
from pathlib import Path


import numpy as np
import pandas as pd
import torch

from nn_laser_stabilizer.data.replay_buffer import ReplayBuffer


def _get_array_dimension(array: np.ndarray) -> int:
    if array.ndim == 0:
        return 1
    elif array.ndim == 1:
        return array.shape[0]
    else:
        raise ValueError(f"Array must be 0D or 1D array, got shape {array.shape}")


def _array_to_tensor(array_like) -> torch.Tensor:
    array = np.asarray(array_like, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    return torch.from_numpy(array.flatten())


def load_buffer_from_csv(
    csv_path: Path,
    extract_transition: Callable[
        [pd.DataFrame, int],
        Optional[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]
    ]
) -> ReplayBuffer:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    num_rows = len(df)
    
    if num_rows < 2:
        raise ValueError(f"CSV file must contain at least 2 rows, got {len(df)}")

    buffer: Optional[ReplayBuffer] = None
    observation_dim: Optional[int] = None
    action_dim: Optional[int] = None

    for idx in range(num_rows):
        transition = extract_transition(df, idx)
        if transition is None:
            continue
        observation, action, reward, next_observation, done = transition

        if buffer is None:
            observation_dim = _get_array_dimension(np.asarray(observation))
            action_dim = _get_array_dimension(np.asarray(action))

            buffer = ReplayBuffer(
                capacity=len(df),
                obs_dim=observation_dim,
                action_dim=action_dim,
            )

        observation_tensor = _array_to_tensor(observation)
        action_tensor = _array_to_tensor(action)
        reward_tensor = torch.tensor(float(reward), dtype=torch.float32).unsqueeze(0)
        next_observation_tensor = _array_to_tensor(next_observation)
        done_tensor = torch.tensor(bool(done), dtype=torch.bool).unsqueeze(0)

        buffer.add(
            observation=observation_tensor,
            action=action_tensor,
            reward=reward_tensor,
            next_observation=next_observation_tensor,
            done=done_tensor,
        )

    if buffer is None:
        raise ValueError(
            "No valid transitions extracted from CSV. "
            "extract_transition(df, idx) returned None for all indices."
        )

    return buffer
