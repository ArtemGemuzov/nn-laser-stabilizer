from typing import Callable, Iterable

import torch
import torch.nn as nn

from nn_laser_stabilizer.rl.agents.optimizer import Optimizer


OptimizerFactory = Callable[[Iterable[torch.nn.Parameter]], Optimizer]


def build_soft_update_pairs(
    *,
    module_pairs: Iterable[tuple[nn.Module, nn.Module]],
) -> list[tuple[nn.Parameter, nn.Parameter]]:
    pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
    for tgt, src in module_pairs:
        for t_param, s_param in zip(tgt.parameters(), src.parameters()):
            pairs.append((t_param, s_param))
    return pairs
