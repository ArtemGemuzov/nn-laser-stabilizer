from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple, Any, Dict
from functools import partial
import time
import traceback

from multiprocessing.connection import Connection

import torch
import torch.multiprocessing as mp

from nn_laser_stabilizer.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.env_wrapper import TorchEnvWrapper
from nn_laser_stabilizer.policy import Policy


@torch.no_grad()
def _warmup_policy(policy: Policy, env: TorchEnvWrapper, num_steps: int = 100) -> None:
    policy.eval()
    for _ in range(num_steps):
        fake_obs = env.observation_space.sample()
        policy.act(fake_obs)


def _policy_factory(policy : Policy):
    return policy.clone()


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
        _warmup_policy(self._policy, self._env)
        
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


class Commands(Enum):
    CLOSE = "close"
    UPDATE_WEIGHTS = "update_weights"
    UPDATE_WEIGHTS_DONE = "update_weights_done"
    ERROR = "error"
    READY = "ready"


@dataclass
class CollectorError:
    type: str
    message: str
    traceback: str


def _collector_worker(
    buffer: ReplayBuffer,
    env_factory: Callable[[], TorchEnvWrapper],
    policy_factory: Callable[[], Policy],
    command_pipe: Connection,
    shared_state_dict: dict,
):
    try:
        policy = policy_factory()
        policy.eval()
        
        env = env_factory()
        obs, _ = env.reset()
        options = {}
        
        _warmup_policy(policy, env)

        command_pipe.send((Commands.READY.value, None))
        
        while True:
            if command_pipe.poll():
                command, _ = command_pipe.recv()
                if command == Commands.UPDATE_WEIGHTS.value:
                    policy.load_state_dict(shared_state_dict)
                    command_pipe.send((Commands.UPDATE_WEIGHTS_DONE.value, None))
                elif command == Commands.CLOSE.value:
                    break
                else:
                    raise ValueError(f"Unknown command received: {command}") 
            
            obs, options = _collect_step(policy, env, obs, buffer, options)
                
    except KeyboardInterrupt:
        pass

    except Exception as e:
        error_info = CollectorError(
            type=type(e).__name__,
            message=str(e),
            traceback=traceback.format_exc(),
        )
        command_pipe.send((Commands.ERROR.value, error_info))
    
    finally:
        if env is not None:
            env.close()


class AsyncCollector:
    def __init__(
        self,
        buffer: ReplayBuffer,
        policy: Policy,
        env_factory: Callable[[], TorchEnvWrapper],
    ):
        self.buffer = buffer
        
        self.env_factory = env_factory
        self.policy = policy
        
        self._parent_pipe, self._child_pipe = mp.Pipe()
        self._process: Optional[mp.Process] = None
        self._running = False
        self._error: Optional[CollectorError] = None
    
    def _send_command(self, command: Commands, data=None) -> None:
        self._parent_pipe.send((command.value, data))
    
    def _worker_has_errors(self) -> bool:
        if self._parent_pipe.poll():
            command, data = self._parent_pipe.recv()
            if command == Commands.ERROR.value:
                self._error = data
                return True
        return False
    
    def _raise_collector_error(self) -> None:  
        if self._error is not None:
            raise RuntimeError(
                f"Collector process encountered an error: {self._error.type}: {self._error.message}\n"
                f"Traceback from collector process:\n{self._error.traceback}"
            )
        else:
            raise RuntimeError("Collector process encountered an error")
         
    def start(self) -> None:
        if self._running:
            raise RuntimeError("Collector is already running")
        
        self.buffer.share_memory()
        self.policy.share_memory()
        
        shared_state_dict = self.policy.state_dict()
        policy_factory = partial(_policy_factory, policy=self.policy)
        
        self._process = mp.Process(
            target=_collector_worker,
            args=(
                self.buffer,
                self.env_factory,
                policy_factory,
                self._child_pipe,
                shared_state_dict,
            ),
            name="DataCollectorWorker",
        )
        self._process.start()

        if not self._process.is_alive():
            raise RuntimeError("Failed to start collector process")

        if not self._parent_pipe.poll(timeout=10):
            raise RuntimeError(f"Collector process did not send {Commands.READY} signal within timeout")
        
        command, data = self._parent_pipe.recv()
        if command == Commands.READY.value:
            self._running = True
        elif command == Commands.ERROR.value:
            self._error = data
            self._raise_collector_error()
        else:
            raise ValueError(f"Unknown command received: {command}") 
    
    def collect(self, num_steps: int, check_interval: float = 0.1) -> None:
        if not self._running:
            raise RuntimeError("Collector is not running")
        
        while len(self.buffer) < num_steps:
            if self._worker_has_errors():
                self._raise_collector_error()
            time.sleep(check_interval)
    
    def sync(self) -> None:
        if not self._running:
            raise RuntimeError("Collector is not running")
        
        if self._worker_has_errors():
            self._raise_collector_error()
        
        self._send_command(Commands.UPDATE_WEIGHTS, None)
        if not self._parent_pipe.poll(timeout=10):
            raise RuntimeError("Collector process did not send UPDATE_WEIGHTS_DONE signal within timeout")
        
        command, data = self._parent_pipe.recv()
        if command == Commands.UPDATE_WEIGHTS_DONE.value:
            pass
        elif command == Commands.ERROR.value:
            self._error = data
            self._raise_collector_error()
        else:
            raise ValueError(f"Unknown command received: {command}")
    
    def stop(self) -> None:
        if not self._running:
            return
        
        self._send_command(Commands.CLOSE, None) 
        
        if self._worker_has_errors():
            self._raise_collector_error()
        
        if self._process is not None:
            self._process.join(timeout=5.0)
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