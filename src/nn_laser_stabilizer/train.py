import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.modules import MLP, ValueOperator, TanhModule, AdditiveGaussianModule
from torchrl.envs import set_exploration_type, ExplorationType, GymEnv, TransformedEnv, DoubleToFloat, Compose, StepCounter
from torchrl.data import ReplayBuffer, BoundedContinuous
from torchrl.objectives import TD3Loss, SoftUpdate

from pid_controller import PIDController
from damped_oscillator import DampedOscillator
from numerical_experimental_setup import NumericalExperimentalSetup

from pid_tuning_experimental_env import PidTuningExperimentalEnv

SETPOINT = 3.0

MASS = 1.0
K = 1.0
C = 0.01

SEED = 42
torch.manual_seed(SEED)

DEVICE = 'cpu'

HIDDEN_SIZE = 64
NUM_LAYERS = 1

pid = PIDController(setpoint=SETPOINT)
oscillator = DampedOscillator(mass=MASS, k=K, c=C)
numerical_model = NumericalExperimentalSetup(oscillator, pid)

base_env = PidTuningExperimentalEnv(numerical_model, device=DEVICE)
env = TransformedEnv(
    base_env,
    Compose(
        DoubleToFloat()
    ),
)
env.set_seed(SEED)

in_keys = ["observation"]
action_spec = env.action_spec_unbatched.to(DEVICE)
actor_net = MLP(
    num_cells=HIDDEN_SIZE,
    depth=NUM_LAYERS,
    out_features=action_spec.shape[-1],
    activation_class=nn.Tanh,
    device=DEVICE,
)

in_keys_actor = in_keys
actor_module = TensorDictModule(
    actor_net,
    in_keys=in_keys_actor,
    out_keys=["param"],
)
actor = TensorDictSequential(
    actor_module,
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
    ),
)

collector = SyncDataCollector(
    env,
    actor_model_explore,
    frames_per_batch=1,
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

batch_size = 64      # Размер батча для обучения
learning_rate = 1e-4   # Скорость обучения
update_target_freq = 1000  # Частота обновления целевых сетей (в шагах)
log_interval = 1     # Частота логирования (в шагах)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_collected_frames = 0

losses = []
rewards = []

kp_log, ki_log, kd_log = [], [], []
x_log, sp_log = [], []

try:
    while True:  
        for tensordict_data in collector:
            x_log.append(oscillator.x)
            sp_log.append(pid.setpoint)

            action = tensordict_data["action"][0].tolist()
            kp_log.append(action[0])
            ki_log.append(action[1])
            kd_log.append(action[2])
            
            buffer.extend(tensordict_data)
            total_collected_frames += tensordict_data.numel()

            current_reward = tensordict_data["next", "reward"].mean().item()
            rewards.append(current_reward)

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size=batch_size).to(DEVICE)

                loss_dict = loss_module(batch)
                loss = loss_dict["loss_actor"] + loss_dict["loss_qvalue"]
                losses.append(loss.item())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if total_collected_frames % update_target_freq == 0:
                    target_net_updater.step()

                if total_collected_frames % log_interval == 0:
                    print(f"Frame {total_collected_frames}, "
                          f"Loss: {np.mean(losses[-log_interval:]):.4f}, "
                          f"Average Reward: {np.mean(rewards[-log_interval:]):.4f}")


except KeyboardInterrupt:
    print("Training interrupted by user.")
    print(f"Total frames collected: {total_collected_frames}")
    print(f"Final Average Loss: {np.mean(losses[-log_interval:]):.4f}")
    print(f"Final Average Reward: {np.mean(rewards[-log_interval:]):.4f}")

    torch.save(model.state_dict(), "td3_model_final.pth")

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

