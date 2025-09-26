from typing import Tuple, Optional

import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MLP, ValueOperator, TanhModule, AdditiveGaussianModule, LSTMModule, get_primers_from_module, set_recurrent_mode
from torchrl.envs import set_exploration_type, ExplorationType
from torchrl.objectives import TD3Loss, SoftUpdate
   

def make_actor_network(config, observation_spec, action_spec) -> TensorDictSequential:
    agent_cfg = config.agent

    modules = []
    actor_mlp = TensorDictModule(
        MLP(
            out_features=action_spec.shape[-1],
            num_cells=agent_cfg.mlp_num_cells,
            activation_class=nn.ReLU
        ),
        in_keys=["observation"],
        out_keys=["param"]
    )
    modules.append(actor_mlp)

    modules.append(
        TanhModule(
            in_keys=["param"],
            out_keys=["action"],
            spec=action_spec,
            clamp=True
        )
    )

    actor = TensorDictSequential(*modules)
    return actor
    
def make_qvalue_network(config, observation_spec, action_spec):
    agent_cfg = config.agent
    modules = []

    qvalue_mlp = TensorDictModule(
        module=MLP(
            out_features=1,
            num_cells=agent_cfg.q_mlp_num_cells,
            activation_class=nn.ReLU,
        ),
        in_keys=["observation", "action"],
        out_keys=["state_action_value"],
    )
    modules.append(qvalue_mlp)

    qvalue = TensorDictSequential(*modules)
    return qvalue


def make_td3_agent(config, observation_spec, action_spec) -> Tuple[nn.ModuleList, TensorDictSequential, AdditiveGaussianModule]:
    actor = make_actor_network(config, observation_spec, action_spec)
    qvalue = make_qvalue_network(config, observation_spec, action_spec)
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
    primers = get_primers_from_module(actor)
    if primers is not None:
        env.append_transform(primers)

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.fake_tensordict()
        for _ in range(100):
            for net in [actor, qvalue]:
                net(td)

def warmup_from_specs(observation_spec, action_spec, actor, qvalue, device="cpu"):
    models = nn.ModuleList([actor, qvalue])

    dummy_observation = torch.zeros(observation_spec.shape, dtype=torch.float32, device=device)
    dummy_action = torch.zeros(action_spec.shape, dtype=torch.float32, device=device)

    td = TensorDict({
        "is_init": torch.tensor(True),
        "observation": dummy_observation,
        "action": dummy_action,
    }, batch_size=[])

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for i in range(100):
            for net in models:
                net(td)

def make_loss_module(config, actor, qvalue, action_spec) -> TD3Loss:
    agent_cfg = config.agent

    # Использование LSTM для Q-function несовместимо с vmap

    loss_module = TD3Loss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=agent_cfg.num_qvalue_nets,
        action_spec=action_spec,
        delay_actor=True,
        delay_qvalue=True,
        noise_clip=agent_cfg.noise_clip,
        policy_noise=agent_cfg.policy_noise,
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
    with set_recurrent_mode(True):
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