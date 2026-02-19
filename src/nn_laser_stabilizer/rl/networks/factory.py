from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.config.types import NetworkType
from nn_laser_stabilizer.rl.networks.base import ActorNetwork, CriticNetwork
from nn_laser_stabilizer.rl.networks.mlp_actor import MLPActorNetwork
from nn_laser_stabilizer.rl.networks.mlp_critic import MLPCriticNetwork
from nn_laser_stabilizer.rl.networks.lstm_actor import LSTMActorNetwork
from nn_laser_stabilizer.rl.networks.lstm_critic import LSTMCriticNetwork


def make_actor_network_from_config(
    network_config: Config,
    obs_dim: int,
    output_dim: int,
) -> ActorNetwork:
    network_type = NetworkType.from_str(network_config.type)
    if network_type == NetworkType.MLP:
        return MLPActorNetwork.from_config(network_config, obs_dim=obs_dim, output_dim=output_dim)
    elif network_type == NetworkType.LSTM:
        return LSTMActorNetwork.from_config(network_config, obs_dim=obs_dim, output_dim=output_dim)
    else:
        raise ValueError(f"Unhandled network type: {network_type}")


def make_critic_network_from_config(
    network_config: Config,
    obs_dim: int,
    action_dim: int,
) -> CriticNetwork:
    network_type = NetworkType.from_str(network_config.type)
    if network_type == NetworkType.MLP:
        return MLPCriticNetwork.from_config(network_config, obs_dim=obs_dim, action_dim=action_dim)
    elif network_type == NetworkType.LSTM:
        return LSTMCriticNetwork.from_config(network_config, obs_dim=obs_dim, action_dim=action_dim)
    else:
        raise ValueError(f"Unhandled network type: {network_type}")
