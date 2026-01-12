import random
import numpy
import torch


def generate_random_seed() -> int:
    return random.randint(0, 2**31 - 1)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)