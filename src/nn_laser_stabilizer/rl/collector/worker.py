from typing import Callable, Optional

import torch.multiprocessing as mp

from nn_laser_stabilizer.rl.data.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.rl.envs.env_wrapper import TorchEnvWrapper
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.collector.connection import CollectorConnection
from nn_laser_stabilizer.rl.collector.utils import CollectorCommand, CollectorWorkerErrorInfo, collect_step, warmup_step


class CollectorWorker: 
    PROCESS_NAME = "DataCollectorWorker"
    
    def __init__(
        self,
        buffer: ReplayBuffer,
        env_factory: Callable[[], TorchEnvWrapper],
        policy_factory: Callable[[], Policy],
        connection: CollectorConnection,
        shared_state_dict: dict,
        seed: Optional[int] = None,
        warmup_steps: int = 0,
    ):
        self.buffer = buffer
        self.env_factory = env_factory
        self.policy_factory = policy_factory
        self.connection = connection
        self.shared_state_dict = shared_state_dict
        self.seed = seed
        self.warmup_steps = warmup_steps
        
        self._process = mp.Process(target=self._run, name=CollectorWorker.PROCESS_NAME)
    
    def start(self) -> None:
        self._process.start()
    
    def is_alive(self) -> bool:
        return self._process.is_alive()
    
    def _run(self) -> None:
        env = None
        try:
            if self.seed is not None:
                from nn_laser_stabilizer.experiment.seed import set_seeds
                set_seeds(self.seed)

            env = self.env_factory()
            
            policy = self.policy_factory()
            policy.eval()
            policy.warmup(env.observation_space)

            observation, options = env.reset()

            self.connection.send_worker_ready()

            for _ in range(self.warmup_steps):
                observation, options = warmup_step(policy, env, observation, options)
            
            while True:
                if self.connection.poll():
                    command, _ = self.connection.recv_command()
                    if command == CollectorCommand.REQUEST_WEIGHT_UPDATE:
                        policy.load_state_dict(self.shared_state_dict)
                        self.connection.send_weight_update_done()
                    elif command == CollectorCommand.SHUTDOWN:
                        self.connection.send_shutdown_complete()
                        break 
                
                observation, options = collect_step(policy, env, observation, self.buffer, options)
                    
        except KeyboardInterrupt:
            pass

        except Exception as e:
            error_info = CollectorWorkerErrorInfo.from_exception(e)
            self.connection.send_worker_error(error_info)
        
        finally:
            if env is not None:
                env.close()

    def stop(self, timeout: Optional[float] = None) -> None:
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join()

