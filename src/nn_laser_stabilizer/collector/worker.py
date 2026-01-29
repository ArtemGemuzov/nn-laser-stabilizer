from typing import Callable, Optional

import torch.multiprocessing as mp

from nn_laser_stabilizer.data.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.envs.env_wrapper import TorchEnvWrapper
from nn_laser_stabilizer.policy.policy import Policy
from nn_laser_stabilizer.collector.connection import CollectorConnection
from nn_laser_stabilizer.collector.utils import CollectorCommand, CollectorWorkerErrorInfo, _collect_step


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
    ):
        self.buffer = buffer
        self.env_factory = env_factory
        self.policy_factory = policy_factory
        self.connection = connection
        self.shared_state_dict = shared_state_dict
        self.seed = seed
        
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

            self.connection.send_worker_ready()

            obs, _ = env.reset()
            options = {}
            
            while True:
                if self.connection.poll():
                    command, _ = self.connection.recv_command()
                    if command == CollectorCommand.REQUEST_WEIGHT_UPDATE:
                        policy.load_state_dict(self.shared_state_dict)
                        self.connection.send_weight_update_done()
                    elif command == CollectorCommand.SHUTDOWN:
                        self.connection.send_shutdown_complete()
                        break 
                
                obs, options = _collect_step(policy, env, obs, self.buffer, options)
                    
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

