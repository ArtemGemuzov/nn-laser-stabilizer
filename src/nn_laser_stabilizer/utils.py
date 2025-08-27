import random
import numpy
import torch

def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(42)