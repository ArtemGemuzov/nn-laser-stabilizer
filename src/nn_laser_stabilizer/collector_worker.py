from typing import Callable, Optional

import torch.multiprocessing as mp

from nn_laser_stabilizer.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.env_wrapper import TorchEnvWrapper
from nn_laser_stabilizer.policy import Policy
from nn_laser_stabilizer.collector_connection import CollectorConnection
from nn_laser_stabilizer.collector_utils import CollectorCommand, CollectorError, _collect_step


class CollectorWorker(mp.Process): 
    def __init__(
        self,
        buffer: ReplayBuffer,
        env_factory: Callable[[], TorchEnvWrapper],
        policy_factory: Callable[[], Policy],
        connection: CollectorConnection,
        shared_state_dict: dict,
        seed: Optional[int] = None,
        name: str = "DataCollectorWorker",
    ):
        super().__init__(name=name)
        self.buffer = buffer
        self.env_factory = env_factory
        self.policy_factory = policy_factory
        self.connection = connection
        self.shared_state_dict = shared_state_dict
        self.seed = seed
    
    def run(self) -> None:
        env = None
        try:
            if self.seed is not None:
                from nn_laser_stabilizer.seed import set_seeds
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
            error_info = CollectorError.from_exception(e)
            self.connection.send_worker_error(error_info)
        
        finally:
            if env is not None:
                env.close()

    def stop(self, timeout: Optional[float] = None) -> None:
        self.join(timeout=timeout)
        if self.is_alive():
            self.terminate()
            self.join()

