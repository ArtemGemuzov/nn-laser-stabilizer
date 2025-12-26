from typing import Optional, Any, Dict, Tuple
from dataclasses import dataclass
import traceback

import torch

from nn_laser_stabilizer.enum_base import BaseEnum
from nn_laser_stabilizer.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.env_wrapper import TorchEnvWrapper
from nn_laser_stabilizer.policy import Policy


class CollectorCommand(BaseEnum):
    WORKER_READY = "worker_ready"
    WORKER_ERROR = "worker_error"
    SHUTDOWN = "shutdown"
    SHUTDOWN_COMPLETE = "shutdown_complete"
    REQUEST_WEIGHT_UPDATE = "request_weight_update"
    WEIGHT_UPDATE_DONE = "weight_update_done"


@dataclass
class CollectorError:
    type: str
    message: str
    traceback: str
    
    @staticmethod
    def from_exception(exception: Exception) -> "CollectorError":
        return CollectorError(
            type=type(exception).__name__,
            message=str(exception),
            traceback=traceback.format_exc(),
        )


def _collect_step(
    policy: Policy,
    env: TorchEnvWrapper,
    obs: torch.Tensor,
    buffer: ReplayBuffer,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    action, options = policy.act(obs, options)
    
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    buffer.add(obs, action, reward, next_obs, done)
    
    if done:
        options = {}
        next_obs, _ = env.reset()
    
    return next_obs, options