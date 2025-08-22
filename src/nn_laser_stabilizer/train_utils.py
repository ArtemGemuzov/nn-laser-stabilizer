from typing import List
import random

import numpy as np
import matplotlib.pyplot as plt

import torch

def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(42)


def plot_results(kp_log: List[float], ki_log: List[float], kd_log: List[float], 
                x_log: List[float], sp_log: List[float]) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(kp_log, label="Kp")
    plt.plot(ki_log, label="Ki")
    plt.plot(kd_log, label="Kd")
    plt.xlabel("Step")
    plt.ylabel("PID coefficients")
    plt.legend()
    plt.title("PID parameters over collected steps")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(x_log, label="x(t)")
    plt.plot(sp_log, linestyle="--", label="setpoint")
    plt.xlabel("Step")
    plt.ylabel("System output")
    plt.legend()
    plt.title("System response")
    plt.tight_layout()
    plt.show()