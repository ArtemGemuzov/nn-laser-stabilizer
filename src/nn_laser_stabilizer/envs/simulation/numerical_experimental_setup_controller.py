from typing import Tuple, Optional
import numpy as np

from nn_laser_stabilizer.envs.experimental_setup_protocol import ExperimentalSetupProtocol


class NumericalExperimentalSetupController(ExperimentalSetupProtocol): 
    def __init__(self):
        pass
    
    def step(self, kp: float, ki: float, kd: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("NumericalExperimentalSetupController.step is not implemented")
    
    def reset(self, kp: float = None, ki: float = None, kd: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("NumericalExperimentalSetupController.reset is not implemented")
    
    def set_seed(self, seed: Optional[int]) -> None:
        raise NotImplementedError("NumericalExperimentalSetupController.set_seed is not implemented")
    
    def close(self) -> None:
        raise NotImplementedError("NumericalExperimentalSetupController.close is not implemented")
    
    @property
    def setpoint(self) -> float:
        raise NotImplementedError("NumericalExperimentalSetupController.setpoint is not implemented")
