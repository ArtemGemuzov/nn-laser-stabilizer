from torchrl.collectors import SyncDataCollector, aSyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, TensorDictReplayBuffer

def make_collector(config, env, actor, replay_buffer):
    collector = aSyncDataCollector(
        env,
        actor,
        frames_per_batch=config.data.frames_per_batch,
        total_frames=config.data.total_frames,
        # replay_buffer=replay_buffer,
        update_at_each_batch=True
    )
    collector.set_seed(config.seed)
    return collector


def make_buffer(config):
    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=config.data.buffer_size)
    )
    return buffer