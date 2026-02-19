import json

import gymnasium as gym

from nn_laser_stabilizer.utils.logger import Logger


class InfoLoggingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, logger: Logger):
        super().__init__(env)
        self._logger = logger

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._logger.log(json.dumps({"event": "step", **info}))
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._logger.log(json.dumps({"event": "reset", **info}))
        return obs, info

    def close(self):
        self.env.close()
        self._logger.close()
