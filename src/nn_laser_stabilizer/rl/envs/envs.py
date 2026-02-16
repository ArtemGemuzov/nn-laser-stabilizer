from nn_laser_stabilizer.rl.envs.base_env import BaseEnv
from nn_laser_stabilizer.rl.envs.neural_controller_delta import NeuralControllerDelta
from nn_laser_stabilizer.rl.envs.neural_controller import NeuralController
from nn_laser_stabilizer.rl.envs.pendulum_no_vel import PendulumNoVel
from nn_laser_stabilizer.rl.envs.pid_delta_tuning import PidDeltaTuning


CUSTOM_ENV_MAP: dict[str, type[BaseEnv]] = {
    "PendulumNoVel": PendulumNoVel,
    "PidDeltaTuning": PidDeltaTuning,
    "NeuralControllerDelta": NeuralControllerDelta,
    "NeuralController": NeuralController,
}