from pathlib import Path
from typing import Any
from dataclasses import dataclass
import traceback

import torch

from nn_laser_stabilizer.utils.enum import BaseEnum
from nn_laser_stabilizer.utils.logger import Logger, NoOpLogger, AsyncFileLogger, SyncFileLogger
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


_NOOP_LOGGER = NoOpLogger()


def make_step_logger(step_logging_config: dict[str, Any] | None) -> Logger:
    if not step_logging_config or not bool(step_logging_config.get("enabled", False)):
        return _NOOP_LOGGER

    log_dir = Path(step_logging_config.get("log_dir", "."))
    log_file = str(step_logging_config.get("log_file", "collector_steps.jsonl"))
    mode = str(step_logging_config.get("mode", "async")).lower()

    if mode == "sync":
        return SyncFileLogger(log_dir=log_dir, log_file=log_file)
    elif mode == "async":
        return AsyncFileLogger(log_dir=log_dir, log_file=log_file)
    else:
        raise ValueError(f"Unknown collector.step_logging.mode: {mode}")


def _env_step(
    policy: Policy,
    env: TorchEnvWrapper,
    obs: torch.Tensor,
    options: dict[str, Any] | None = None,
    step_logger: Logger = _NOOP_LOGGER,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if options is None:
        options = {}

    action, options = policy.act(obs, options)
    policy_info = options.pop("policy_info")

    next_obs, reward, terminated, truncated, step_info = env.step(action)
    done = terminated or truncated

    step_logger.log_dict(
        {
            "event": "step",
            "policy_info": policy_info,
            "env_info": step_info,
        }
    )

    if done:
        next_obs, reset_info = env.reset()
        reset_info = dict(reset_info)
        step_logger.log_dict(
            {
                "event": "reset",
                "env_info": reset_info,
            }
        )
        options = reset_info
    else:
        options.update(step_info)

    return obs, action, reward, next_obs, done, options


def warmup_step(
    policy: Policy,
    env: TorchEnvWrapper,
    obs: torch.Tensor,
    options: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _, _, _, next_obs, _, options = _env_step(
        policy=policy,
        env=env,
        obs=obs,
        options=options,
    )
    return next_obs, options


def collect_step(
    policy: Policy,
    env: TorchEnvWrapper,
    obs: torch.Tensor,
    buffer: ReplayBuffer,
    options: dict[str, Any] | None = None,
    step_logger: Logger = _NOOP_LOGGER,
) -> tuple[torch.Tensor, dict[str, Any]]:
    obs, action, reward, next_obs, done, options = _env_step(
        policy=policy,
        env=env,
        obs=obs,
        options=options,
        step_logger=step_logger,
    )
    buffer.add(obs, action, reward, next_obs, done)
    return next_obs, options