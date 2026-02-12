from typing import Callable, Optional, Any, Dict
from abc import ABC, abstractmethod
from functools import partial
import time

import torch

from nn_laser_stabilizer.rl.data.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.rl.envs.env_wrapper import TorchEnvWrapper
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.collector.worker import CollectorWorker
from nn_laser_stabilizer.rl.collector.connection import CollectorConnection
from nn_laser_stabilizer.rl.collector.utils import collect_step, CollectorWorkerError


class BaseCollector(ABC):
    def __init__(self, buffer: ReplayBuffer):
        self.buffer: ReplayBuffer = buffer
        self._running = False

    def start(self) -> None:
        self._check_not_running()
        self._on_start()
        self._running = True

    @abstractmethod
    def _on_start(self) -> None:
        ...

    @abstractmethod
    def ensure(self, min_size: int) -> None:
        ...

    @abstractmethod
    def collect(self, num_steps: int) -> None:
        ...

    def sync(self) -> None:
        pass

    def stop(self) -> None:
        if not self._running:
            return
        self._on_stop()
        self._running = False

    @abstractmethod
    def _on_stop(self) -> None:
        ...

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _check_running(self) -> None:
        if not self._running:
            raise RuntimeError("Collector is not running")

    def _check_not_running(self) -> None:
        if self._running:
            raise RuntimeError("Collector is already running")


def _policy_factory(policy: Policy):
    return policy.clone()


class SyncCollector(BaseCollector):
    def __init__(
        self,
        buffer: ReplayBuffer,
        env: TorchEnvWrapper,
        policy: Policy,
    ):
        super().__init__(buffer)
        self._env: TorchEnvWrapper = env
        self._policy: Policy = policy

        self._current_observation: Optional[torch.Tensor] = None
        self._options: Dict[str, Any] = {}

    def _on_start(self) -> None:
        self._policy.eval()
        self._policy.warmup(self._env.observation_space)

        self._current_observation, self._options = self._env.reset()
        assert self._current_observation is not None

    def ensure(self, min_size: int) -> None:
        self._check_running()
        assert self._current_observation is not None
        while len(self.buffer) < min_size:
            self._current_observation, self._options = collect_step(
                self._policy,
                self._env,
                self._current_observation,
                self.buffer,
                self._options,
            )

    def collect(self, num_steps: int) -> None:
        self._check_running()
        assert self._current_observation is not None
        for _ in range(num_steps):
            self._current_observation, self._options = collect_step(
                self._policy,
                self._env,
                self._current_observation,
                self.buffer,
                self._options,
            )

    def _on_stop(self) -> None:
        if self._env is not None:
            self._env.close()


class AsyncCollector(BaseCollector):
    READY_TIMEOUT_SEC = 600.0
    WEIGHT_UPDATE_DONE_TIMEOUT_SEC = 10.0
    PROCESS_JOIN_TIMEOUT_SEC = 5.0

    def __init__(
        self,
        buffer: ReplayBuffer,
        policy: Policy,
        env_factory: Callable[[], TorchEnvWrapper],
        seed: Optional[int] = None,
        check_interval: float = 0.1,
    ):
        super().__init__(buffer)

        self._env_factory = env_factory
        self._policy: Policy = policy
        self._seed = seed
        self._check_interval = check_interval

        self._connection, self._child_connection = CollectorConnection.create_pair()

        self._process: Optional[CollectorWorker] = None

    def _on_start(self) -> None:
        self.buffer.share_memory()
        self._policy.share_memory()

        shared_state_dict = self._policy.state_dict()
        policy_factory = partial(_policy_factory, policy=self._policy)

        self._process = CollectorWorker(
            buffer=self.buffer,
            env_factory=self._env_factory,
            policy_factory=policy_factory,
            connection=self._child_connection,
            shared_state_dict=shared_state_dict,
            seed=self._seed,
        )
        self._process.start()

        if not self._process.is_alive():
            raise RuntimeError("Failed to start collector process")

        self._connection.wait_for_ready(timeout=AsyncCollector.READY_TIMEOUT_SEC)

    def ensure(self, min_size: int) -> None:
        self._check_running()
        
        while len(self.buffer) < min_size:
            self._connection.poll_worker_error()
            time.sleep(self._check_interval)

    def collect(self, num_steps: int) -> None:
        self._check_running()

        if num_steps <= 0:
            return
        self.ensure(len(self.buffer) + num_steps)

    def sync(self) -> None:
        self._check_running()

        self._connection.poll_worker_error()

        self._connection.request_weight_update()
        self._connection.wait_for_weight_update_done(
            timeout=AsyncCollector.WEIGHT_UPDATE_DONE_TIMEOUT_SEC
        )

    def _on_stop(self) -> None:
        self._shutdown(wait_for_shutdown=True)

    def _shutdown(self, wait_for_shutdown: bool) -> None:
        self._connection.poll_worker_error()

        self._connection.send_shutdown()
        if wait_for_shutdown:
            self._connection.wait_for_shutdown_complete(
                timeout=AsyncCollector.PROCESS_JOIN_TIMEOUT_SEC
            )

        if self._process is not None:
            self._process.stop(timeout=AsyncCollector.PROCESS_JOIN_TIMEOUT_SEC)
            self._process = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._running:
            return
        if exc_type is KeyboardInterrupt or CollectorWorkerError:
            self._shutdown(wait_for_shutdown=False)
        else:
            self._shutdown(wait_for_shutdown=True)
        self._running = False


def make_collector_from_config(
    collector_config,
    env_factory: Callable[[], TorchEnvWrapper],
    buffer: ReplayBuffer,
    policy: Policy,
    seed: Optional[int] = None,
) -> BaseCollector:
    if collector_config.is_async:
        return AsyncCollector(
            buffer=buffer,
            policy=policy,
            env_factory=env_factory,
            seed=seed,
        )
    else:
        env = env_factory()
        return SyncCollector(
            buffer=buffer,
            env=env,
            policy=policy,
        )
