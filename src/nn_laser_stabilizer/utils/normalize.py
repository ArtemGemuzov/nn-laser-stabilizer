import numpy as np


def normalize_to_minus1_plus1(value: float, min_val: float, max_val: float) -> float:
    span = max_val - min_val
    norm_01 = (value - min_val) / span
    return 2.0 * norm_01 - 1.0


def normalize_to_01(value: float, min_val: float, max_val: float) -> float:
    span = max_val - min_val
    norm_01 = (value - min_val) / span
    return float(np.clip(norm_01, 0.0, 1.0))


def denormalize_from_minus1_plus1(value: float, min_val: float, max_val: float) -> float:
    norm_01 = (value + 1.0) / 2.0
    span = max_val - min_val
    return min_val + norm_01 * span
