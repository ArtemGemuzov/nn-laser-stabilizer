from nn_laser_stabilizer.envs.nn_pid_env import NNPid
from nn_laser_stabilizer.envs.pendulum_no_vel_env import PendulumNoVelEnv
from nn_laser_stabilizer.envs.pid_delta_tuning_env import PidDeltaTuningEnv


CUSTOM_ENV_MAP: dict[str, type] = {
    "PendulumNoVelEnv": PendulumNoVelEnv,
    "PidDeltaTuningEnv": PidDeltaTuningEnv,
    "NNPid": NNPid,
}