"""Оценка размера сетей/буфера и времени инференса для конфига neural_controller.

Запуск (из conda-окружения nn-laser-stabilizer):
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --config neural_controller --iters 20000
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from nn_laser_stabilizer.config.config import find_and_load_config
from nn_laser_stabilizer.rl.algorithms.factory import build_agent
from nn_laser_stabilizer.rl.data.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.rl.envs.factory import get_spaces_from_config


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000.0:
            return f"{n:.2f} {unit}"
        n /= 1000.0
    return f"{n:.2f} TB"


def measure(fn, iters: int, warmup: int) -> np.ndarray:
    """Замеряет время вызова fn() iters раз, возвращает массив времён в мкс."""
    for _ in range(warmup):
        fn()
    times_ns = np.empty(iters, dtype=np.int64)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        times_ns[i] = time.perf_counter_ns() - t0
    return times_ns / 1000.0


def print_timing(title: str, us: np.ndarray) -> None:
    print(f"  {title}")
    print(f"    среднее / std    : {us.mean():.2f} / {us.std():.2f} мкс")
    print(f"    медиана          : {np.median(us):.2f} мкс")
    print(f"    p90 / p99        : {np.percentile(us, 90):.2f} / {np.percentile(us, 99):.2f} мкс")
    print(f"    min / max        : {us.min():.2f} / {us.max():.2f} мкс")
    print(f"    пропускная способн.: {1e6 / us.mean():,.0f} шагов/с")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="neural_controller",
                        help="имя конфига внутри configs/")
    parser.add_argument("--iters", type=int, default=200_000)
    parser.add_argument("--warmup", type=int, default=1000)
    args = parser.parse_args()

    torch.manual_seed(0)

    config = find_and_load_config(args.config)
    capacity = int(config.buffer.capacity)

    observation_space, action_space = get_spaces_from_config(
        config.env, seed=config.get("seed", None)
    )
    obs_dim = observation_space.dim
    action_dim = action_space.dim

    agent = build_agent(
        algorithm_config=config.algorithm,
        observation_space=observation_space,
        action_space=action_space,
    )
    policy = agent.default_policy()

    bytes_per_param = 4  # float32
    hidden_repr = config.get("algorithm.actor.network.mlp_hidden_sizes", "?")

    modules = {
        name.lstrip("_"): value
        for name, value in vars(agent).items()
        if isinstance(value, nn.Module)
    }

    print(f"\nКонфиг: {args.config}\n")

    print("=" * 66)
    print("РАЗМЕР НЕЙРОННЫХ СЕТЕЙ")
    print("=" * 66)
    print(f"  алгоритм           : {type(agent).__name__}")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}, hidden={hidden_repr}")
    print(f"  сетей всего        : {len(modules)}")
    total = 0
    actor_params = 0
    for name, module in modules.items():
        p = sum(t.numel() for t in module.parameters())
        total += p
        if name == "actor":
            actor_params = p
        print(f"    {name:<18}: {p:>10,} парам.  {fmt_bytes(p * bytes_per_param)}")
    print(f"  ИТОГО параметров   : {total:>10,} парам.  {fmt_bytes(total * bytes_per_param)}")
    if actor_params:
        print()
        print(f"  Инференс использует только actor:")
        print(f"                       {actor_params:>10,} парам.  {fmt_bytes(actor_params * bytes_per_param)}")
    print()

    buffer = ReplayBuffer(capacity=capacity, obs_dim=obs_dim, action_dim=action_dim)
    buffer_tensors = {
        "observations": buffer.observations,
        "actions": buffer.actions,
        "rewards": buffer.rewards,
        "next_observations": buffer.next_observations,
        "dones": buffer.dones,
    }
    print("=" * 66)
    print("РАЗМЕР БУФЕРА ВОСПРОИЗВЕДЕНИЯ")
    print("=" * 66)
    print(f"  capacity           : {capacity:,} переходов")
    total_buffer = 0
    for name, t in buffer_tensors.items():
        nbytes = t.element_size() * t.nelement()
        total_buffer += nbytes
        print(f"    {name:<18}: {fmt_bytes(nbytes)}  ({tuple(t.shape)}, {t.dtype})")
    print(f"  ИТОГО буфер        : {fmt_bytes(total_buffer)}")
    print()

    obs = torch.randn(1, obs_dim, dtype=torch.float32)
    actor = modules["actor"]

    def bare_mlp():
        with torch.no_grad():
            return actor._network(obs)

    def deploy():
        with torch.no_grad():
            out = actor._network(obs)
            mean = out.output.chunk(2, dim=-1)[0]
            return actor._scaler(mean)

    def full_act():
        policy.act(obs, {})

    print("=" * 66)
    print(f"ВРЕМЯ ИНФЕРЕНСА")
    print("=" * 66)
    print_timing("Прямой проход MLP:",
                 measure(bare_mlp, args.iters, args.warmup))
    print()
    print_timing("Путь развёртывания (mean + tanh + scale):",
                 measure(deploy, args.iters, args.warmup))
    print()
    print_timing("Полный policy.act (сэмплирование + log_prob + логирование):",
                 measure(full_act, args.iters, args.warmup))
    print()


if __name__ == "__main__":
    main()
