from typing import Any

import torch

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.utils.pid import PIDDelta
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.exploration.base_exploaration import BaseExplorationPolicy


class PIDExplorationPolicy(BaseExplorationPolicy):
    CUR_ERROR_KEY = "error"
    PREV_ERROR_KEY = "prev_error"
    PREV_PREV_ERROR_KEY = "prev_prev_error"

    def __init__(
        self,
        inner: Policy,
        action_space: Box,
        *,
        start_step: int,
        end_step: int | None,
        pid: PIDDelta,
        max_delta: float,
    ):
        super().__init__(inner, action_space, start_step=start_step, end_step=end_step)
        self._pid = pid
        self._max_delta = max_delta

    def _explore(self, action: torch.Tensor, options: dict[str, Any]) -> torch.Tensor:
        cur_error = float(options[self.CUR_ERROR_KEY])
        prev_error = float(options[self.PREV_ERROR_KEY])
        prev_prev_error = float(options[self.PREV_PREV_ERROR_KEY])
        
        delta = self._pid.compute_from_errors(cur_error, prev_error, prev_prev_error)
        action_value = torch.tensor([delta / self._max_delta], dtype=torch.float32)
        return torch.clamp(action_value, -1.0, 1.0)

    def clone(self) -> "PIDExplorationPolicy":
        pid = PIDDelta(kp=self._pid.kp, ki=self._pid.ki, kd=self._pid.kd, dt=self._pid.dt)
        return PIDExplorationPolicy(
            inner=self._inner.clone(),
            action_space=self._action_space,
            start_step=self._start_step,
            end_step=self._end_step,
            pid=pid,
            max_delta=self._max_delta,
        )

    @classmethod
    def from_config(
        cls, exploration_config: Config, *, policy: Policy, action_space: Box,
    ) -> "PIDExplorationPolicy":
        start_step = int(exploration_config.get("start_step", 0))
        steps = int(exploration_config.get("steps", 0))
        end_step_raw = exploration_config.get("end_step", None)
        end_step = None if end_step_raw is None else int(end_step_raw)
        
        kp = float(exploration_config.kp)
        ki = float(exploration_config.ki)
        kd = float(exploration_config.kd)
        dt = float(exploration_config.dt)
        max_delta = float(exploration_config.max_delta)

        if start_step < 0:
            raise ValueError("exploration.start_step must be >= 0 for PID exploration")
        if steps < 0:
            raise ValueError("exploration.steps must be >= 0 for PID exploration")
        if end_step is None:
            end_step = start_step + steps
        if end_step < start_step:
            raise ValueError("exploration.end_step must be >= exploration.start_step for PID exploration")
        if dt <= 0.0:
            raise ValueError("exploration.dt must be > 0")
        if max_delta <= 0.0:
            raise ValueError("exploration.max_delta must be > 0")

        pid = PIDDelta(kp=kp, ki=ki, kd=kd, dt=dt)
        return cls(
            inner=policy,
            action_space=action_space,
            start_step=start_step,
            end_step=end_step,
            pid=pid,
            max_delta=max_delta,
        )
