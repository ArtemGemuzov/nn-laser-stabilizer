from typing import Any
from dataclasses import dataclass
import traceback

import torch

from nn_laser_stabilizer.enum import BaseEnum
from nn_laser_stabilizer.data.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.envs.env_wrapper import TorchEnvWrapper
from nn_laser_stabilizer.policy.policy import Policy


class CollectorCommand(BaseEnum):
    WORKER_READY = "worker_ready"
    WORKER_ERROR = "worker_error"
    SHUTDOWN = "shutdown"
    SHUTDOWN_COMPLETE = "shutdown_complete"
    REQUEST_WEIGHT_UPDATE = "request_weight_update"
    WEIGHT_UPDATE_DONE = "weight_update_done"


@dataclass
class CollectorWorkerErrorInfo:
    type: str
    message: str
    traceback: str
    
    @staticmethod
    def from_exception(exception: Exception) -> "CollectorWorkerErrorInfo":
        return CollectorWorkerErrorInfo(
            type=type(exception).__name__,
            message=str(exception),
            traceback=traceback.format_exc(),
        )
    
    def raise_exception(self) -> None:
        raise CollectorWorkerError(self)


class CollectorWorkerError(Exception):
    def __init__(self, error: CollectorWorkerErrorInfo):
        message = (
            f"Collector worker process encountered an error: {error.type}: {error.message}\n"
            f"Traceback from worker process:\n{error.traceback}"
        )
        super().__init__(message)


def collect_step(
    policy: Policy,
    env: TorchEnvWrapper,
    obs: torch.Tensor,
    buffer: ReplayBuffer,
    options: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if options is None:
        options = {}
    action, options = policy.act(obs, options)
    
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    buffer.add(obs, action, reward, next_obs, done)
    
    if done:
        options = {}
        next_obs, _ = env.reset()
    
    return next_obs, options