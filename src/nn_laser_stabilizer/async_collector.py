from enum import Enum
from typing import Callable, Optional

import torch
import torch.multiprocessing as mp

from nn_laser_stabilizer.replay_buffer import SharedReplayBuffer
from nn_laser_stabilizer.env import TorchEnvWrapper


class Commands(str, Enum):
    CLOSE = "close"
    UPDATE_WEIGHTS = "update_weights"


def _collector_worker(
    buffer: SharedReplayBuffer,
    env_factory: Callable[[], TorchEnvWrapper],
    policy_factory: Callable[[], torch.nn.Module],
    command_pipe: mp.connection.PipeConnection,
):
    env = env_factory()
    policy = policy_factory()
    policy.eval()
    
    obs, _ = env.reset()
    running = True
    
    try:
        while running:
            if command_pipe.poll():
                command, data = command_pipe.recv()
                if command == Commands.CLOSE.value:
                    running = False
                    break
                elif command == Commands.UPDATE_WEIGHTS.value:
                    policy.load_state_dict(data)
            
            with torch.no_grad():
                action = policy(obs)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            buffer.add(obs, action, reward, next_obs, done)
            
            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs
                
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


class AsyncCollector:
    def __init__(
        self,
        buffer: SharedReplayBuffer,
        env_factory: Callable[[], TorchEnvWrapper],
        policy_factory: Callable[[], torch.nn.Module],
    ):
        self.buffer = buffer
        self.env_factory = env_factory
        self.policy_factory = policy_factory
        
        self._parent_pipe, self._child_pipe = mp.Pipe()
        self._process: Optional[mp.Process] = None
        self._running = False
    
    def _send_command(self, command: Commands, data=None) -> None:
        try:
            self._parent_pipe.send((command.value, data))
        except (BrokenPipeError, OSError):
            pass
    
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
            daemon=True,
        )
        self._process.start()
        self._running = True
    
    def synchronize(self, policy: torch.nn.Module) -> None:
        if not self._running:
            raise RuntimeError("Collector is not running")
        
        state_dict = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        self._send_command(Commands.UPDATE_WEIGHTS, state_dict)
    
    def stop(self, timeout: Optional[float] = 5.0) -> None:
        if not self._running:
            return
        
        self._send_command(Commands.CLOSE, None) 
        
        if self._process is not None:
            self._process.join(timeout=timeout)
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
