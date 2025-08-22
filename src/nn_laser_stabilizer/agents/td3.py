from typing import Tuple, Optional

import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MLP, ValueOperator, TanhModule, AdditiveGaussianModule, LSTMModule, get_primers_from_module
from torchrl.envs import set_exploration_type, ExplorationType, TransformedEnv
from torchrl.objectives import TD3Loss, SoftUpdate

def make_actor_network(config, action_spec) -> TensorDictSequential:
    actor_lstm = LSTMModule(
        input_size=3, # TODO убрать
        hidden_size=64,
        num_layers=2,
        in_key="observation",
        out_key="param"
    )

    actor_mlp = TensorDictModule(
        MLP(
            out_features=3 # TODO убрать
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
    qvalue_net = MLP(
        out_features=1,
        depth=2,
        num_cells=64,
    )

    qvalue = ValueOperator(
        in_keys=["action", "observation"],
        module=qvalue_net,
    )
    return qvalue


def make_td3_agent(action_spec, config) -> Tuple[nn.ModuleList, TensorDictSequential, AdditiveGaussianModule]:
    actor = make_actor_network(config, action_spec)
    qvalue = make_qvalue_network(config)
    return actor, qvalue

def add_exploration(actor, action_spec, config):
    exploration_module = AdditiveGaussianModule(
        spec=action_spec,
        annealing_num_steps=config.exploration_annealing_steps
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


def make_loss_module(actor, qvalue, action_spec, config) -> TD3Loss:
    loss_module = TD3Loss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=config.num_qvalue_nets,
        action_spec=action_spec
    )
    loss_module.make_value_estimator(gamma=config.gamma)
    return loss_module


def make_optimizers(loss_module: TD3Loss, config) -> Tuple[torch.optim.Adam, torch.optim.Adam]:
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = torch.optim.Adam(actor_params, lr=config.learning_rate)
    optimizer_critic = torch.optim.Adam(critic_params, lr=config.learning_rate)
    
    return optimizer_actor, optimizer_critic


def make_target_updater(loss_module: TD3Loss, config) -> SoftUpdate:
    return SoftUpdate(loss_module, eps=config.target_update_eps)

def train_step(batch, loss_module: TD3Loss, optimizer_actor: torch.optim.Adam, 
               optimizer_critic: torch.optim.Adam, target_net_updater: SoftUpdate, 
               step_idx: int, config) -> Tuple[float, Optional[float]]:
    loss_qvalue, _ = loss_module.value_loss(batch)
    loss_qvalue.backward()
    optimizer_critic.step()
    optimizer_critic.zero_grad(set_to_none=True)

    loss_qvalue_val = loss_qvalue.detach().item()
    loss_actor_val = None

    if step_idx % config.update_target_freq == 0:
        loss_actor, _ = loss_module.actor_loss(batch)
        loss_actor.backward()
        optimizer_actor.step()
        optimizer_actor.zero_grad(set_to_none=True)
        
        target_net_updater.step()
        loss_actor_val = loss_actor.detach().item()
    
    return loss_qvalue_val, loss_actor_val