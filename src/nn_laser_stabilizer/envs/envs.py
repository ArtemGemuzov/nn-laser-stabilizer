from nn_laser_stabilizer.envs.neural_pid_env import NeuralPIDEnv
from nn_laser_stabilizer.envs.pendulum_no_vel_env import PendulumNoVelEnv
from nn_laser_stabilizer.envs.pid_delta_tuning_env import PidDeltaTuningEnv


CUSTOM_ENV_MAP: dict[str, type] = {
    "PendulumNoVelEnv": PendulumNoVelEnv,
    "PidDeltaTuningEnv": PidDeltaTuningEnv,
    "NeuralPIDEnv": NeuralPIDEnv,
}