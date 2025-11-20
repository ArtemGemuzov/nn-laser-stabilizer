from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
import time
import traceback

from multiprocessing.connection import Connection

import torch
import torch.multiprocessing as mp

from nn_laser_stabilizer.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.env import TorchEnvWrapper
from nn_laser_stabilizer.policy import Policy


def _collect_step(
    policy: Policy,
    env: TorchEnvWrapper,
    obs: torch.Tensor,
    buffer: ReplayBuffer,
) -> torch.Tensor:
    action = policy.act(obs)
    
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    buffer.add(obs, action, reward, next_obs, done)
    
    if done:
        next_obs, _ = env.reset()
    
    return next_obs


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
        self._running = False
    
    def start(self) -> None:
        if self._running:
            raise RuntimeError("Collector is already running")
        
        self._policy.eval()
        self._cur_obs, _ = self._env.reset()
        self._running = True
    
    def collect(self, num_steps: int) -> None:
        if not self._running:
            raise RuntimeError("Collector is not running")
        
        for _ in range(num_steps):
            self._cur_obs = _collect_step(
                self._policy,
                self._env,
                self._cur_obs,
                self.buffer,
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
):
    try:
        policy = policy_factory()
        policy.eval()
        
        env = env_factory()
        obs, _ = env.reset()
        
        # TODO: надо получить fake_obs и проверить работу, перед запуском обучения

        command_pipe.send((Commands.READY.value, None))
        
        while True:
            if command_pipe.poll():
                command, data = command_pipe.recv()
                if command == Commands.UPDATE_WEIGHTS.value:
                    policy.load_state_dict(data)
                elif command == Commands.CLOSE.value:
                    break
                else:
                    raise ValueError(f"Unknown command received: {command}") 
            
            obs = _collect_step(policy, env, obs, buffer)
                
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
        env_factory: Callable[[], TorchEnvWrapper],
        policy_factory: Callable[[], Policy],
    ):
        self.buffer = buffer
        self.env_factory = env_factory
        self.policy_factory = policy_factory
        
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
        
        self._process = mp.Process(
            target=_collector_worker,
            args=(
                self.buffer,
                self.env_factory,
                self.policy_factory,
                self._child_pipe,
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
    
    def sync(self, policy: Policy) -> None:
        if not self._running:
            raise RuntimeError("Collector is not running")
        
        if self._worker_has_errors():
            self._raise_collector_error()
        
        state_dict = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        self._send_command(Commands.UPDATE_WEIGHTS, state_dict)
    
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