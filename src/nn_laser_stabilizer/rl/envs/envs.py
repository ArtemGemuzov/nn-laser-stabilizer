from nn_laser_stabilizer.rl.envs.base_env import BaseEnv
from nn_laser_stabilizer.rl.envs.neural_pid_delta_env import NeuralControllerDeltaEnv
from nn_laser_stabilizer.rl.envs.neural_controller_env import NeuralControllerEnv
from nn_laser_stabilizer.rl.envs.pendulum_no_vel_env import PendulumNoVelEnv
from nn_laser_stabilizer.rl.envs.pid_delta_tuning_env import PidDeltaTuningEnv


CUSTOM_ENV_MAP: dict[str, type[BaseEnv]] = {
    "PendulumNoVelEnv": PendulumNoVelEnv,
    "PidDeltaTuningEnv": PidDeltaTuningEnv,
    "NeuralControllerDeltaEnv": NeuralControllerDeltaEnv,
    "NeuralControllerEnv": NeuralControllerEnv,
}