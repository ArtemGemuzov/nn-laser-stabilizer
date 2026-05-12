from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.utils.enum import BaseEnum
from nn_laser_stabilizer.rl.algorithms.base import Agent
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.envs.spaces.discrete import Discrete
from nn_laser_stabilizer.rl.algorithms.td3.agent import TD3Agent
from nn_laser_stabilizer.rl.algorithms.td3bc.agent import TD3BCAgent
from nn_laser_stabilizer.rl.algorithms.bc.agent import BCAgent
from nn_laser_stabilizer.rl.algorithms.sac.agent import SACAgent
from nn_laser_stabilizer.rl.algorithms.cql.agent import CQLAgent
from nn_laser_stabilizer.rl.algorithms.dqn.agent import DQNAgent


class AlgorithmType(BaseEnum):
    TD3 = "td3"
    TD3BC = "td3bc"
    BC = "bc"
    SAC = "sac"
    CQL = "cql"
    DQN = "dqn"


def build_agent(
    algorithm_config: Config,
    observation_space: Box,
    action_space: Box | Discrete,
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

    elif algorithm_type == AlgorithmType.CQL:
        return CQLAgent.from_config(algorithm_config, observation_space, action_space)

    elif algorithm_type == AlgorithmType.DQN:
        if not isinstance(action_space, Discrete):
            raise ValueError("DQN requires a discrete action space. Add 'discrete_action' wrapper to the env config.")
        return DQNAgent.from_config(algorithm_config, observation_space, action_space)

    else:
        raise ValueError(f"Unhandled algorithm type: {algorithm_type}")
