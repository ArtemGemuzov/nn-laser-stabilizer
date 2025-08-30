import random
import numpy as np
import pandas as pd

import torch
from tensorboard.backend.event_processing import event_accumulator

def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(42)

def tensorboard_to_df(logdir : str):
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    tags = ea.Tags()["scalars"]

    all_data = []

    for tag in tags:
        events = ea.Scalars(tag)
        for e in events:
            all_data.append({
                "wall_time": e.wall_time,
                "step": e.step,
                "tag": tag,
                "value": e.value
            })

    return pd.DataFrame(all_data)

