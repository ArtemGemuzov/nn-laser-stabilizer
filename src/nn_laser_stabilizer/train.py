import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.modules import MLP, ValueOperator, TanhModule, AdditiveGaussianModule, LSTMModule
from torchrl.envs import set_exploration_type, ExplorationType, TransformedEnv, DoubleToFloat, Compose, ObservationNorm, InitTracker, StepCounter
from torchrl.data import ReplayBuffer, BoundedContinuous
from torchrl.objectives import TD3Loss, SoftUpdate

from pid_controller import PIDController
from oscillator import Oscillator
from numerical_experimental_setup import NumericalExperimentalSetup

from pid_tuning_experimental_env import PidTuningExperimentalEnv

SETPOINT = 10000.0

MASS = 1.0
K = 1.0
C = 0.01
NOISE_LEVEL = 0.0

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = 'cpu'

HIDDEN_SIZE = 128
NUM_LAYERS = 2

pid = PIDController(setpoint=SETPOINT)
oscillator = Oscillator(mass=MASS, k=K, c=C)
numerical_model = NumericalExperimentalSetup(oscillator, pid)

base_env = PidTuningExperimentalEnv(numerical_model, device=DEVICE)
env = TransformedEnv(
    base_env,
    Compose(
        StepCounter(),
        InitTracker(),
        DoubleToFloat()
    ),
)
env.set_seed(SEED)
_ = env.reset()

in_keys = ["observation"]
action_spec = env.action_spec_unbatched.to(DEVICE)
actor_lstm = LSTMModule(
    input_size=3,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    in_key="observation",
    out_key="param",
    device=DEVICE,
)
env.append_transform(actor_lstm.make_tensordict_primer())

actor_mlp = MLP(
    out_features=3,
    activation_class=nn.Tanh,
    device=DEVICE,
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

qvalue_net = MLP(
    num_cells=HIDDEN_SIZE,
    depth=NUM_LAYERS,
    out_features=1,
    activation_class=nn.ReLU,
    device=DEVICE,
)

qvalue = ValueOperator(
    in_keys=["action"] + in_keys,
    module=qvalue_net,
)

model = nn.ModuleList([actor, qvalue])

with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
    td = env.fake_tensordict().to(DEVICE)
    for net in model:
        net(td)

actor_model_explore = TensorDictSequential(
    actor,
    AdditiveGaussianModule(
        spec=action_spec,
        device=DEVICE,
        annealing_num_steps=10_000
    ),
)

collector = SyncDataCollector(
    env,
    actor_model_explore,
    frames_per_batch=50,
    total_frames=-1,
    device=DEVICE,
)
collector.set_seed(SEED)

buffer = ReplayBuffer()

loss_module = TD3Loss(
    actor_network=model[0],
    qvalue_network=model[1],
    num_qvalue_nets=2,
    action_spec=action_spec
)
loss_module.make_value_estimator(gamma=0.9)

target_net_updater = SoftUpdate(loss_module, eps=0.995)

batch_size = 256      # Размер батча для обучения
learning_rate = 1e-5   # Скорость обучения
update_target_freq = 2  # Частота обновления целевых сетей (в шагах)
log_interval = 1     # Частота логирования (в шагах)

critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
actor_params = list(loss_module.actor_network_params.flatten_keys().values())

optimizer_actor = torch.optim.Adam(
    actor_params,
    lr=learning_rate
)
optimizer_critic = torch.optim.Adam(
    critic_params,
    lr=learning_rate,
)

total_collected_frames = 0

losses = []
rewards = []

kp_log, ki_log, kd_log = [], [], []
x_log, sp_log = [], []

n_opt = 1

try: 
    for tensordict_data in collector:
        observation = tensordict_data["observation"]

        x_log.extend(observation[:, 0].tolist())
        sp_log.extend(observation[:, 2].tolist())

        action = tensordict_data["action"]
        kp_log.extend(action[:, 0].tolist())
        ki_log.extend(action[:, 1].tolist())
        kd_log.extend(action[:, 2].tolist())
        
        buffer.extend(tensordict_data)
        total_collected_frames += tensordict_data.numel()

        current_reward = tensordict_data["next", "reward"].mean().item()
        rewards.append(current_reward)

        if len(buffer) >= batch_size:
            for _ in range(n_opt):
                batch = buffer.sample(batch_size=batch_size).to(DEVICE)

                loss_qvalue = loss_module(batch)["loss_qvalue"]

                loss_qvalue.backward()
                optimizer_critic.step()
                optimizer_critic.zero_grad(set_to_none=True)
                
                if total_collected_frames % update_target_freq == 0:
                    loss_actor = loss_module(batch)["loss_actor"]

                    loss_actor.backward()
                    optimizer_actor.step()
                    optimizer_actor.zero_grad(set_to_none=True)

                    target_net_updater.step()
                
                with torch.no_grad():
                    current_loss_dict = loss_module(batch)
                    loss_actor_val = current_loss_dict["loss_actor"].item()
                    loss_qvalue_val = current_loss_dict["loss_qvalue"].item()
                    losses.append(loss_actor_val + loss_qvalue_val)

        if total_collected_frames % log_interval == 0:
            print(f"Frame {total_collected_frames}, "
                    f"Loss: {np.mean(losses[-log_interval:]):.4f}, "
                    f"Average Reward: {np.mean(rewards[-log_interval:]):.4f}")


except KeyboardInterrupt:
    print("Training interrupted by user.")
    print(f"Total frames collected: {total_collected_frames}")
    print(f"Final Average Loss: {np.mean(losses[-log_interval:]):.4f}")
    print(f"Final Average Reward: {np.mean(rewards[-log_interval:]):.4f}")

    collector.shutdown()

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

