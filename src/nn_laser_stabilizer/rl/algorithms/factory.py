from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.config.types import AlgorithmType
from nn_laser_stabilizer.rl.algorithms.base import Agent, Learner
from nn_laser_stabilizer.rl.envs.spaces.box import Box

from nn_laser_stabilizer.rl.algorithms.td3.agent import TD3Agent
from nn_laser_stabilizer.rl.algorithms.td3.loss import TD3Loss
from nn_laser_stabilizer.rl.algorithms.td3.learner import TD3Learner

from nn_laser_stabilizer.rl.algorithms.td3bc.loss import TD3BCLoss

from nn_laser_stabilizer.rl.algorithms.bc.agent import BCAgent
from nn_laser_stabilizer.rl.algorithms.bc.loss import BCLoss
from nn_laser_stabilizer.rl.algorithms.bc.learner import BCLearner

from nn_laser_stabilizer.rl.algorithms.sac.agent import SACAgent
from nn_laser_stabilizer.rl.algorithms.sac.loss import SACLoss
from nn_laser_stabilizer.rl.algorithms.sac.learner import SACLearner


def build_algorithm(
    algorithm_config: Config,
    observation_space: Box,
    action_space: Box,
) -> tuple[Agent, Learner]:
    algorithm_type = AlgorithmType.from_str(algorithm_config.type)

    if algorithm_type == AlgorithmType.TD3:
        agent = TD3Agent.from_config(algorithm_config, observation_space, action_space)
        loss = TD3Loss.from_config(algorithm_config, agent)
        learner = TD3Learner.from_config(algorithm_config, agent, loss)
        return agent, learner

    elif algorithm_type == AlgorithmType.TD3BC:
        agent = TD3Agent.from_config(algorithm_config, observation_space, action_space)
        loss = TD3BCLoss.from_config(algorithm_config, agent)
        learner = TD3Learner.from_config(algorithm_config, agent, loss)
        return agent, learner

    elif algorithm_type == AlgorithmType.BC:
        agent = BCAgent.from_config(algorithm_config, observation_space, action_space)
        loss = BCLoss.from_config(algorithm_config, agent)
        learner = BCLearner.from_config(algorithm_config, agent, loss)
        return agent, learner

    elif algorithm_type == AlgorithmType.SAC:
        agent = SACAgent.from_config(algorithm_config, observation_space, action_space)
        loss = SACLoss.from_config(algorithm_config, agent)
        learner = SACLearner.from_config(algorithm_config, agent, loss)
        return agent, learner

    else:
        raise ValueError(f"Unhandled algorithm type: {algorithm_type}")
