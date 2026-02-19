from typing import Any
from dataclasses import dataclass
import traceback

import torch

from nn_laser_stabilizer.utils.enum import BaseEnum
from nn_laser_stabilizer.rl.data.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.rl.envs.torch_wrapper import TorchEnvWrapper
from nn_laser_stabilizer.rl.policy.policy import Policy


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


def _env_step(
    policy: Policy,
    env: TorchEnvWrapper,
    obs: torch.Tensor,
    options: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if options is None:
        options = {}
    action, options = policy.act(obs, options)

    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
        options = {}
        next_obs, info = env.reset()

    options.update(info)
    return obs, action, reward, next_obs, done, options


def warmup_step(
    policy: Policy,
    env: TorchEnvWrapper,
    obs: torch.Tensor,
    options: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _, _, _, next_obs, _, options = _env_step(policy, env, obs, options)
    return next_obs, options


def collect_step(
    policy: Policy,
    env: TorchEnvWrapper,
    obs: torch.Tensor,
    buffer: ReplayBuffer,
    options: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    obs, action, reward, next_obs, done, options = _env_step(policy, env, obs, options)
    buffer.add(obs, action, reward, next_obs, done)
    return next_obs, options