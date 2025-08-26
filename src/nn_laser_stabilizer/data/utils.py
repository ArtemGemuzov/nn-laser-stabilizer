from torchrl.collectors import SyncDataCollector, aSyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

def make_sync_collector(config, env, actor):
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=config.data.frames_per_batch,
        total_frames=config.data.total_frames,
    )
    collector.set_seed(config.seed)
    return collector

def make_async_collector(config, make_env_fn, actor, replay_buffer):
    collector = aSyncDataCollector(
        create_env_fn=make_env_fn,  
        policy=actor,
        frames_per_batch=config.data.frames_per_batch,
        total_frames=config.data.total_frames,
        replay_buffer=replay_buffer,
        update_at_each_batch=True,
    )
    collector.set_seed(config.seed)
    return collector

def make_buffer(config):
    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=config.data.buffer_size)
    )
    return buffer