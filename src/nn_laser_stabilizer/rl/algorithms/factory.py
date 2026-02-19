from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.config.types import AlgorithmType
from nn_laser_stabilizer.rl.algorithms.base import Agent
from nn_laser_stabilizer.rl.envs.spaces.box import Box

from nn_laser_stabilizer.rl.algorithms.td3.agent import TD3Agent
from nn_laser_stabilizer.rl.algorithms.td3bc.agent import TD3BCAgent
from nn_laser_stabilizer.rl.algorithms.bc.agent import BCAgent
from nn_laser_stabilizer.rl.algorithms.sac.agent import SACAgent


def build_agent(
    algorithm_config: Config,
    observation_space: Box,
    action_space: Box,
) -> Agent:
    algorithm_type = AlgorithmType.from_str(algorithm_config.type)

    if algorithm_type == AlgorithmType.TD3:
        return TD3Agent.from_config(algorithm_config, observation_space, action_space)

    elif algorithm_type == AlgorithmType.TD3BC:
        return TD3BCAgent.from_config(algorithm_config, observation_space, action_space)

    elif algorithm_type == AlgorithmType.BC:
        return BCAgent.from_config(algorithm_config, observation_space, action_space)

    elif algorithm_type == AlgorithmType.SAC:
        return SACAgent.from_config(algorithm_config, observation_space, action_space)

    else:
        raise ValueError(f"Unhandled algorithm type: {algorithm_type}")
