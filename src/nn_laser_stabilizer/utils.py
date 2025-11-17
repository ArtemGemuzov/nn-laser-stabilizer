import os
import random

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(42)

def get_hydra_runtime_output_dir() -> str:
    try:
        from hydra.core.hydra_config import HydraConfig
        return HydraConfig.get().runtime.output_dir
    except Exception:
        return os.getcwd()