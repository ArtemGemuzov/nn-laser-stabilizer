import os
import time
from logging import getLogger

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from nn_laser_stabilizer.agents import (
    make_td3_agent,
    warmup_from_specs,
)
from nn_laser_stabilizer.config import find_configs_dir, get_hydra_output_dir
from nn_laser_stabilizer.envs import make_env, make_specs
from nn_laser_stabilizer.logging import set_seeds
from nn_laser_stabilizer.training import make_async_collector, make_buffer

logger = getLogger(__name__)

CONFIG_NAME = "inference_async"
ACTOR_CHECKPOINT = "scripts/weights/actor.pth"
QVALUE_CHECKPOINT = "scripts/weights/qvalue.pth"


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(get_original_cwd(), path)


def _load_pretrained_models(actor: torch.nn.Module, qvalue: torch.nn.Module) -> None:
    actor_path = _resolve_path(ACTOR_CHECKPOINT)
    qvalue_path = _resolve_path(QVALUE_CHECKPOINT)

    if not os.path.isfile(actor_path):
        raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")
    if not os.path.isfile(qvalue_path):
        raise FileNotFoundError(f"Critic checkpoint not found: {qvalue_path}")

    actor_state = torch.load(actor_path, map_location="cpu")
    qvalue_state = torch.load(qvalue_path, map_location="cpu")

    actor.load_state_dict(actor_state)
    qvalue.load_state_dict(qvalue_state)

    logger.info("Pretrained actor and critic weights loaded.")


@hydra.main(config_path=find_configs_dir(), config_name=CONFIG_NAME, version_base=None)
def main(config: DictConfig) -> None:
    set_seeds(config.seed)

    if not config.env.get("setpoint"):
        config.env.setpoint = float(input("Enter setpoint for environment: "))
    logger.info("Setpoint = %s", config.env.setpoint)

    output_dir = get_hydra_output_dir()
    config.output_dir = output_dir

    # aSyncDataCollector внутри себя создает фейковое окружение, поэтому при первом вызове нужно вернуть симуляцию окружения
    def make_env_factory(config):
        first_call = True

        def env_fn():
            nonlocal first_call
            if not first_call:
                return make_env(config)
            
            first_call = False
            from copy import deepcopy
            config_for_mocked_env = deepcopy(config)
            if not config_for_mocked_env.serial.get("use_mock"):
                config_for_mocked_env.serial.use_mock = True
            return make_env(config_for_mocked_env)

        return env_fn

    make_env_fn = make_env_factory(config)

    specs = make_specs(config.env.bounds)
    action_spec = specs["action"]
    observation_spec = specs["observation"]

    actor, qvalue = make_td3_agent(config, observation_spec, action_spec)
    warmup_from_specs(observation_spec, action_spec, actor, qvalue)
    _load_pretrained_models(actor, qvalue)

    buffer = make_buffer(config)
    collector = make_async_collector(config, make_env_fn, actor, buffer)

    try:
        logger.info("Inference process started.")
        collector.start()

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.warning("Inference interrupted by user.")

    except Exception as ex:
        logger.warning(f"Error while inference: {ex}")

    finally:
        logger.info("Inference finished.")
        logger.info(f"Final buffer size: {len(buffer)} samples")

        model_save_dir = get_hydra_output_dir("saved_models")

        actor_path = os.path.join(model_save_dir, "actor.pth")
        torch.save(actor.state_dict(), actor_path)

        qvalue_path = os.path.join(model_save_dir, "qvalue.pth")
        torch.save(qvalue.state_dict(), qvalue_path)
        
        buffer_save_dir = get_hydra_output_dir("saved_data")
        buffer_path = os.path.join(buffer_save_dir, "replay_buffer.pkl")
        os.makedirs(buffer_save_dir, exist_ok=True)
        
        logger.info(f"Saving replay buffer with {len(buffer)} samples to {buffer_path}")
        buffer.dump(buffer_path)
        logger.info("Replay buffer saved successfully.")
        
        collector.async_shutdown()


if __name__ == "__main__":
    main()