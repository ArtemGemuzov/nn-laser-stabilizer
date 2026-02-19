import gymnasium as gym

from nn_laser_stabilizer.utils.time import CallIntervalTracker


class StepTrackingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, time_multiplier: float = 1e6):
        super().__init__(env)
        self._step_count = 0
        self._interval_tracker = CallIntervalTracker(time_multiplier=time_multiplier)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        info["step"] = self._step_count
        info["step_interval_us"] = self._interval_tracker.tick()
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        self._interval_tracker.reset()
        return obs, info
