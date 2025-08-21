import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.modules import MLP, ValueOperator, TanhModule, AdditiveGaussianModule, LSTMModule, get_primers_from_module
from torchrl.envs import set_exploration_type, ExplorationType, TransformedEnv, DoubleToFloat, Compose, InitTracker
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.objectives import TD3Loss, SoftUpdate

from nn_laser_stabilizer.pid_controller import PIDController
from nn_laser_stabilizer.oscillator import DuffingOscillator
from nn_laser_stabilizer.numerical_experimental_setup import NumericalExperimentalSetup
from nn_laser_stabilizer.pid_tuning_experimental_env import PidTuningExperimentalEnv


@dataclass
class Config:
    setpoint: float
    mass: float
    k_linear: float
    k_nonlinear: float
    k_damping: float
    process_noise_std: float
    measurement_noise_std: float
    hidden_size: int
    num_layers: int
    input_size: int
    output_size: int
    seed: int
    device: str
    batch_size: int
    learning_rate: float
    update_target_freq: int
    gamma: float
    target_update_eps: float
    frames_per_batch: int
    total_frames: int
    buffer_size: int
    update_to_data: int
    max_train_steps: int
    exploration_annealing_steps: int
    qvalue_num_cells: int
    qvalue_depth: int
    qvalue_out_features: int
    num_qvalue_nets: int


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_env(config: Config) -> TransformedEnv:
    pid = PIDController(setpoint=config.setpoint)
    oscillator = DuffingOscillator(
        mass=config.mass, 
        k_linear=config.k_linear, 
        k_nonlinear=config.k_nonlinear, 
        k_damping=config.k_damping,
        process_noise_std=config.process_noise_std, 
        measurement_noise_std=config.measurement_noise_std
    )
    numerical_model = NumericalExperimentalSetup(oscillator, pid)

    base_env = PidTuningExperimentalEnv(numerical_model, device=config.device)
    env = TransformedEnv(
        base_env,
        Compose(
            InitTracker(),
            DoubleToFloat()
        )
    )
    env.set_seed(config.seed)
    return env


def make_actor_network(action_spec, config: Config) -> TensorDictSequential:
    actor_lstm = LSTMModule(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        in_key="observation",
        out_key="param",
        device=config.device,
    )

    actor_mlp = MLP(
        out_features=config.output_size,
        activation_class=nn.Tanh,
        device=config.device,
    )
    actor_mlp = TensorDictModule(actor_mlp, in_keys=["param"], out_keys=["param"])

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


def make_qvalue_network(in_keys: List[str], config: Config) -> ValueOperator:
    qvalue_net = MLP(
        num_cells=config.qvalue_num_cells,
        depth=config.qvalue_depth,
        out_features=config.qvalue_out_features,
        activation_class=nn.ReLU,
        device=config.device,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )
    return qvalue


def make_td3_agent(env: TransformedEnv, config: Config) -> Tuple[nn.ModuleList, TensorDictSequential, AdditiveGaussianModule]:
    in_keys = ["observation"]
    action_spec = env.action_spec_unbatched.to(config.device)
    
    actor = make_actor_network(action_spec, config)
    qvalue = make_qvalue_network(in_keys, config)
    model = nn.ModuleList([actor, qvalue])

    primers = get_primers_from_module(model)
    env.append_transform(primers)

    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = env.fake_tensordict().to(config.device)
        for net in model:
            net(td)

    exploration_module = AdditiveGaussianModule(
        spec=action_spec,
        device=config.device,
        annealing_num_steps=config.exploration_annealing_steps
    )

    actor_model_explore = TensorDictSequential(
        actor,
        exploration_module
    )

    return model, actor_model_explore, exploration_module


def make_collector(env: TransformedEnv, actor_model_explore: TensorDictSequential, config: Config) -> SyncDataCollector:
    collector = SyncDataCollector(
        env,
        actor_model_explore,
        frames_per_batch=config.frames_per_batch,
        total_frames=config.total_frames,
        device=config.device,
    )
    collector.set_seed(config.seed)
    return collector


def make_buffer(config: Config) -> ReplayBuffer:
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=config.buffer_size)
    )
    return buffer


def make_loss_module(model: nn.ModuleList, action_spec, config: Config) -> TD3Loss:
    loss_module = TD3Loss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=config.num_qvalue_nets,
        action_spec=action_spec
    )
    loss_module.make_value_estimator(gamma=config.gamma)
    return loss_module


def make_optimizers(loss_module: TD3Loss, config: Config) -> Tuple[torch.optim.Adam, torch.optim.Adam]:
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = torch.optim.Adam(actor_params, lr=config.learning_rate)
    optimizer_critic = torch.optim.Adam(critic_params, lr=config.learning_rate)
    
    return optimizer_actor, optimizer_critic


def make_target_updater(loss_module: TD3Loss, config: Config) -> SoftUpdate:
    return SoftUpdate(loss_module, eps=config.target_update_eps)


def train_step(batch, loss_module: TD3Loss, optimizer_actor: torch.optim.Adam, 
               optimizer_critic: torch.optim.Adam, target_net_updater: SoftUpdate, 
               step_idx: int, config: Config) -> Tuple[float, Optional[float]]:
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


def plot_results(kp_log: List[float], ki_log: List[float], kd_log: List[float], 
                x_log: List[float], sp_log: List[float]) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(kp_log, label="Kp")
    plt.plot(ki_log, label="Ki")
    plt.plot(kd_log, label="Kd")
    plt.xlabel("Step")
    plt.ylabel("PID coefficients")
    plt.legend()
    plt.title("PID parameters over collected steps")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(x_log, label="x(t)")
    plt.plot(sp_log, linestyle="--", label="setpoint")
    plt.xlabel("Step")
    plt.ylabel("System output")
    plt.legend()
    plt.title("System response")
    plt.tight_layout()
    plt.show()