from tensordict.nn import TensorDictSequential
from torchrl.envs import TransformedEnv
from torchrl.collectors import SyncDataCollector, aSyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage

def make_collector(env: TransformedEnv, actor_model_explore: TensorDictSequential, config):
    collector = aSyncDataCollector(
        env,
        actor_model_explore,
        frames_per_batch=config.frames_per_batch,
        total_frames=config.total_frames,
    )
    collector.set_seed(config.seed)
    return collector


def make_buffer(config) -> ReplayBuffer:
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=config.buffer_size)
    )
    return buffer