import os
import time
import logging
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch

from nn_laser_stabilizer.utils import (
    set_seeds
)
from nn_laser_stabilizer.agents.td3 import (
    make_td3_agent,
    make_loss_module,
    make_optimizers,
    make_target_updater,
    train_step,
    warmup_from_specs
)
from nn_laser_stabilizer.envs.utils import make_simulated_env, make_specs, add_logger_to_env

from nn_laser_stabilizer.data.utils import make_buffer, make_async_collector, make_sync_collector
from nn_laser_stabilizer.config.find_configs_dir import find_configs_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@hydra.main(config_path=find_configs_dir(), config_name="train_simulation", version_base=None)
def main(config: DictConfig) -> None:
    set_seeds(config.seed)

    hydra_output_dir = HydraConfig.get().runtime.output_dir
    env_log_dir = os.path.join(hydra_output_dir, "env_logs")
    os.makedirs(env_log_dir, exist_ok=True)

    def make_env_with_logger():
        env = make_simulated_env(config)
        return add_logger_to_env(env, env_log_dir)

    def make_env_no_logger():
        return make_simulated_env(config)

    specs = make_specs(config.env.bounds)
    action_spec = specs["action"]
    observation_spec = specs["observation"]

    actor, qvalue = make_td3_agent(config, observation_spec, action_spec)
    warmup_from_specs(observation_spec, action_spec, actor, qvalue)

    buffer = make_buffer(config)
    collector = make_sync_collector(config, make_env_no_logger(), actor)

    loss_module = make_loss_module(config, actor, qvalue, action_spec)
    optimizer_actor, optimizer_critic = make_optimizers(config, loss_module)
    target_net_updater = make_target_updater(config, loss_module)

    train_config = config.train

    logger.info("Benchmark started")

    n_steps = 10_000

    # ===== С логером =====
    env_logger = make_env_with_logger()
    observation = env_logger.reset()
    step_times_logger = []

    for _ in range(n_steps):
        action = env_logger.rand_action(observation)
        start_step = time.perf_counter()
        observation = env_logger.step(action)
        step_times_logger.append(time.perf_counter() - start_step)

    step_times_logger = np.array(step_times_logger)
    mean_logger = step_times_logger.mean()
    std_logger = step_times_logger.std()

    logger.info(f"[With logger] Time per step: mean={mean_logger:.8f} sec, std={std_logger:.8f} sec")

    # ===== Без логера =====
    env_no_logger = make_env_no_logger()
    observation = env_no_logger.reset()
    step_times_no_logger = []

    for _ in range(n_steps):
        action = env_no_logger.rand_action(observation)
        start_step = time.perf_counter()
        observation = env_no_logger.step(action)
        step_times_no_logger.append(time.perf_counter() - start_step)

    step_times_no_logger = np.array(step_times_no_logger)
    mean_no_logger = step_times_no_logger.mean()
    std_no_logger = step_times_no_logger.std()

    logger.info(f"[No logger] Time per step: mean={mean_no_logger:.8f} sec, std={std_no_logger:.8f} sec")

    # ===== Разница =====
    n_steps = 20_000
    mean_diff = mean_logger - mean_no_logger

    logger.info(f"Difference (logger - no logger): mean={mean_diff:.8f} sec")

    # ==== Policy ====
    obs_tensor = env_no_logger.fake_tensordict()

    policy_times = []

    for _ in range(n_steps):
        start_policy = time.perf_counter()
        with torch.no_grad():  
            action_tensor = actor(obs_tensor)
        policy_times.append(time.perf_counter() - start_policy)

    policy_times = np.array(policy_times)
    mean_policy = policy_times.mean()
    std_policy = policy_times.std()

    logger.info(f"[Policy only] Time per action: mean={mean_policy:.8f} sec, std={std_policy:.8f} sec")

    # ==== Collector ====
    n_steps = 100
    collector_iter = iter(collector)

    traj_times = []

    for _ in range(n_steps):  
        start_traj = time.perf_counter()
        
        trajectory = next(collector_iter)
        buffer.add(trajectory.unsqueeze(0).to_tensordict())
        
        traj_times.append(time.perf_counter() - start_traj)

    traj_times = np.array(traj_times)
    mean_traj_time = traj_times.mean()
    std_traj_time = traj_times.std()

    mean_frame_time = mean_traj_time / trajectory.numel()
    std_frame_time = std_traj_time / trajectory.numel()

    logger.info(
        f"[Collector trajectory] Time per trajectory: mean={mean_traj_time:.8f} sec, std={std_traj_time:.8f} sec;\n"
        f"time per frame: mean={mean_frame_time:.8f} sec, std={std_frame_time:.8f} sec"
    )

    n_train_steps = 100 
    batch_size = config.train.batch_size

    # ===== С обновлением актера =====
    train_times_actor = []

    for _ in range(n_train_steps):
        batch = buffer.sample(batch_size)
        start = time.perf_counter()
        train_step(batch, loss_module, optimizer_actor, optimizer_critic, target_net_updater, update_actor=True)
        train_times_actor.append(time.perf_counter() - start)

    train_times_actor = np.array(train_times_actor)
    mean_actor_time = train_times_actor.mean()
    std_actor_time = train_times_actor.std()

    mean_actor_per_sample = mean_actor_time / batch_size
    std_actor_per_sample = std_actor_time / batch_size

    logger.info(
        f"[Training step] With actor update: mean={mean_actor_time:.8f} sec, std={std_actor_time:.8f} sec;\n"
        f"per sample: mean={mean_actor_per_sample:.8f} sec, std={std_actor_per_sample:.8f} sec"
    )

    # ===== Без обновления актера =====
    train_times_no_actor = []

    for _ in range(n_train_steps):
        batch = buffer.sample(batch_size)
        start = time.perf_counter()
        train_step(batch, loss_module, optimizer_actor, optimizer_critic, target_net_updater, update_actor=False)
        train_times_no_actor.append(time.perf_counter() - start)

    train_times_no_actor = np.array(train_times_no_actor)
    mean_no_actor_time = train_times_no_actor.mean()
    std_no_actor_time = train_times_no_actor.std()

    mean_no_actor_per_sample = mean_no_actor_time / batch_size
    std_no_actor_per_sample = std_no_actor_time / batch_size

    logger.info(
        f"[Training step] No actor update: mean={mean_no_actor_time:.8f} sec, std={std_no_actor_time:.8f} sec;\n"
        f"per sample: mean={mean_no_actor_per_sample:.8f} sec, std={std_no_actor_per_sample:.8f} sec"
    )

    # ===== Время полного цикла обновлений =====
    n_updates = train_config.update_to_data
    batch_size = train_config.batch_size
    update_actor_freq = train_config.update_actor_freq

    step_times = []

    start = time.perf_counter()
    for i in range(n_updates):
        batch = buffer.sample(batch_size)
        update_actor = i % update_actor_freq == 0
        step_start = time.perf_counter()
        train_step(batch, loss_module, optimizer_actor, optimizer_critic, target_net_updater, update_actor)
        step_times.append(time.perf_counter() - step_start)
    full_cycle_time = time.perf_counter() - start

    step_times = np.array(step_times)
    mean_step_time = step_times.mean()
    std_step_time = step_times.std()

    mean_per_sample = mean_step_time / batch_size
    std_per_sample = std_step_time / batch_size

    logger.info(
        f"[Full training cycle] Total time for {n_updates} steps: {full_cycle_time:.6f} sec;\n"
        f"mean per step: {mean_step_time:.8f} sec, std per step: {std_step_time:.8f} sec;\n"
        f"mean per sample: {mean_per_sample:.8f} sec, std per sample: {std_per_sample:.8f} sec"
    )

    logger.info("Benchmark completed")


if __name__ == "__main__":
    main()
