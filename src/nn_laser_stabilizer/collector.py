from typing import Callable, Optional, Any, Dict
from functools import partial
import time

import torch
import torch.multiprocessing as mp

from nn_laser_stabilizer.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.env_wrapper import TorchEnvWrapper
from nn_laser_stabilizer.policy import Policy
from nn_laser_stabilizer.collector_worker import CollectorWorker
from nn_laser_stabilizer.collector_connection import CollectorConnection
from nn_laser_stabilizer.collector_utils import CollectorError, _collect_step


def _policy_factory(policy : Policy):
    return policy.clone()


class SyncCollector:
    def __init__(
        self,
        buffer: ReplayBuffer,
        env: TorchEnvWrapper,
        policy: Policy,
    ):
        self.buffer = buffer
        self._env = env
        self._policy = policy
        
        self._cur_obs: Optional[torch.Tensor] = None
        self._options: Dict[str, Any] = {}
        self._running = False
    
    def start(self) -> None:
        if self._running:
            raise RuntimeError("Collector is already running")
        
        self._policy.eval()
        self._policy.warmup(self._env.observation_space)
        
        self._cur_obs, _ = self._env.reset()
        self._options = {}
        self._running = True
    
    def collect(self, num_steps: int) -> None:
        if not self._running:
            raise RuntimeError("Collector is not running")
        
        for _ in range(num_steps):
            self._cur_obs, self._options = _collect_step(
                self._policy,
                self._env,
                self._cur_obs,
                self.buffer,
                self._options,
            )
    
    def stop(self) -> None:
        if not self._running:
            return
        
        if self._env is not None:
            self._env.close()
            self._env = None
        
        self._policy = None
        self._cur_obs = None
        self._running = False
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def __del__(self):
        if self._env is not None:
            self.stop()


class AsyncCollector:
    READY_TIMEOUT_SEC = 600.0
    WEIGHT_UPDATE_DONE_TIMEOUT_SEC = 10.0
    PROCESS_JOIN_TIMEOUT_SEC = 5.0
    
    def __init__(
        self,
        buffer: ReplayBuffer,
        policy: Policy,
        env_factory: Callable[[], TorchEnvWrapper],
        seed: Optional[int] = None,
    ):
        self.buffer = buffer
        
        self.env_factory = env_factory
        self.policy = policy
        self.seed = seed
        
        self._connection, self._child_connection = CollectorConnection.create_pair()
        
        self._process: Optional[mp.Process] = None
        self._running = False
         
    def start(self) -> None:
        if self._running:
            raise RuntimeError("Collector is already running")
        
        self.buffer.share_memory()
        self.policy.share_memory()
        
        shared_state_dict = self.policy.state_dict()
        policy_factory = partial(_policy_factory, policy=self.policy)
        
        self._process = CollectorWorker(
            buffer=self.buffer,
            env_factory=self.env_factory,
            policy_factory=policy_factory,
            connection=self._child_connection,
            shared_state_dict=shared_state_dict,
            seed=self.seed,
            name="DataCollectorWorker",
        )
        self._process.start()

        if not self._process.is_alive():
            raise RuntimeError("Failed to start collector process")

        error = self._connection.wait_for_ready(timeout=AsyncCollector.READY_TIMEOUT_SEC)
        if error is not None:
            self._raise_worker_error(error)
        
        self._running = True 
    
    def collect(self, num_steps: int, check_interval: float = 0.1) -> None:
        if not self._running:
            raise RuntimeError("Collector is not running")
        
        while len(self.buffer) < num_steps:
            error = self._connection.poll_worker_error()
            if error is not None:
                self._raise_worker_error(error)
            time.sleep(check_interval)
    
    def sync(self) -> None:
        if not self._running:
            raise RuntimeError("Collector is not running")
        
        error = self._connection.poll_worker_error()
        if error is not None:
            self._raise_worker_error(error)
        
        self._connection.request_weight_update()
        error = self._connection.wait_for_weight_update_done(timeout=AsyncCollector.WEIGHT_UPDATE_DONE_TIMEOUT_SEC)
        if error is not None:
            self._raise_worker_error(error)
    
    def stop(self) -> None:
        if not self._running:
            return
        
        self._connection.send_shutdown()
        
        error = self._connection.wait_for_shutdown_complete(timeout=AsyncCollector.PROCESS_JOIN_TIMEOUT_SEC)
        if error is not None:
            self._raise_worker_error(error)
        
        if self._process is not None:
            self._process.join(timeout=AsyncCollector.PROCESS_JOIN_TIMEOUT_SEC)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join()
            self._process = None
        
        self._running = False
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def __del__(self):
        if self._running:
            self.stop()

    def _raise_worker_error(self, error: CollectorError) -> None:
        raise RuntimeError(
            f"Collector process encountered an error: {error.type}: {error.message}\n"
            f"Traceback from collector process:\n{error.traceback}"
        )