from typing import Tuple, Optional

import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MLP, ValueOperator, TanhModule, AdditiveGaussianModule, LSTMModule, get_primers_from_module
from torchrl.envs import set_exploration_type, ExplorationType
from torchrl.objectives import TD3Loss, SoftUpdate

def make_actor_network(config, observation_spec, action_spec) -> TensorDictSequential:
    agent_cfg = config.agent

    actor_lstm = LSTMModule(
        input_size=observation_spec.shape[-1],
        hidden_size=agent_cfg.lstm_hidden_size,
        num_layers=agent_cfg.lstm_num_layers,
        in_key="observation",
        out_key="param"
    )

    actor_mlp = TensorDictModule(
        MLP(
            out_features=action_spec.shape[-1],
            depth=agent_cfg.mlp_depth,
            num_cells=agent_cfg.mlp_num_cells
        ), 
        in_keys=["param"], 
        out_keys=["param"])

    actor = TensorDictSequential(
        actor_lstm,
        actor_mlp,
        TanhModule(
            in_keys=["param"],
            out_keys=["action"],
            spec=action_spec,
        ),
    )
    return actor


def make_qvalue_network(config) -> ValueOperator:
    agent_cfg = config.agent

    qvalue_net = MLP(
        out_features=1,
        depth=agent_cfg.q_mlp_depth,
        num_cells=agent_cfg.q_mlp_num_cells,
    )

    qvalue = ValueOperator(
        in_keys=["action", "observation"],
        module=qvalue_net,
    )
    return qvalue


def make_td3_agent(config, observation_spec, action_spec) -> Tuple[nn.ModuleList, TensorDictSequential, AdditiveGaussianModule]:
    actor = make_actor_network(config, observation_spec, action_spec)
    qvalue = make_qvalue_network(config)
    return actor, qvalue

def add_exploration(config, actor, action_spec):
    agent_cfg = config.agent

    exploration_module = AdditiveGaussianModule(
        spec=action_spec,
        sigma_init=agent_cfg.exploration_sigma_init,
        sigma_end=agent_cfg.exploration_sigma_end,
        annealing_num_steps=agent_cfg.exploration_annealing_steps
    )

    actor_model_explore = TensorDictSequential(
        actor,
        exploration_module
    )
    return actor_model_explore, exploration_module


def warmup(env, actor, qvalue):
    models = nn.ModuleList([actor, qvalue])

    primers = get_primers_from_module(models)
    env.append_transform(primers)

    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = env.fake_tensordict()
        for net in models:
            net(td)


def make_loss_module(config, actor, qvalue, action_spec) -> TD3Loss:
    agent_cfg = config.agent

    loss_module = TD3Loss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=agent_cfg.num_qvalue_nets,
        action_spec=action_spec,
        delay_actor=True,
        delay_qvalue=True
    )
    loss_module.make_value_estimator(gamma=agent_cfg.gamma)
    return loss_module


def make_optimizers(config, loss_module: TD3Loss) -> Tuple[torch.optim.Adam, torch.optim.Adam]:
    agent_cfg = config.agent

    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = torch.optim.Adam(actor_params, lr=agent_cfg.learning_rate_actor)
    optimizer_critic = torch.optim.Adam(critic_params, lr=agent_cfg.learning_rate_critic)
    
    return optimizer_actor, optimizer_critic


def make_target_updater(config, loss_module: TD3Loss) -> SoftUpdate:
    return SoftUpdate(loss_module, eps=config.agent.target_update_eps)


def train_step(batch, loss_module: TD3Loss, optimizer_actor: torch.optim.Adam, 
               optimizer_critic: torch.optim.Adam, target_net_updater: SoftUpdate, update_actor : bool) -> Tuple[float, Optional[float]]:
    loss_qvalue, _ = loss_module.value_loss(batch)
    loss_qvalue.backward()
    optimizer_critic.step()
    optimizer_critic.zero_grad(set_to_none=True)

    loss_qvalue_val = loss_qvalue.detach().item()
    loss_actor_val = None

    if update_actor:
        loss_actor, _ = loss_module.actor_loss(batch)
        loss_actor.backward()
        optimizer_actor.step()
        optimizer_actor.zero_grad(set_to_none=True)
        
        target_net_updater.step()
        loss_actor_val = loss_actor.detach().item()
    
    return loss_qvalue_val, loss_actor_val