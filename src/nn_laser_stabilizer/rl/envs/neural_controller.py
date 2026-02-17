import json
from enum import Enum
from functools import partial
from typing import Optional

import numpy as np
import gymnasium as gym

from nn_laser_stabilizer.logger import AsyncFileLogger, Logger
from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.envs.base_env import BaseEnv
from nn_laser_stabilizer.rl.envs.bounded_value import BoundedValue
from nn_laser_stabilizer.rl.envs.arx_plant_backend import ARXPlantBackend
from nn_laser_stabilizer.rl.envs.plant_backend import ExperimentalPlantBackend, PlantBackend
from nn_laser_stabilizer.normalize import denormalize_from_minus1_plus1
from nn_laser_stabilizer.time import CallIntervalTracker


class ActionType(Enum):
    DELTA = "delta"
    ABSOLUTE = "absolute"


class NeuralController(BaseEnv):
    LOG_SOURCE = "NeuralControllerDelta"

    def __init__(
        self,
        *,
        backend: PlantBackend,
        base_logger: Logger,
        control_min: int,
        control_max: int,
        process_variable_max: int,
        reset_value: int,
        reset_steps: int,
        observe_prev_error: bool,
        observe_prev_prev_error: bool,
        observe_control_output: bool,
        action_type: ActionType,
        max_action_delta: int = 0,
        action_penalty: float = 0.0,
    ):
        super().__init__()

        self._action_type = action_type
        self._action_penalty = action_penalty

        if action_penalty > 0 and action_type != ActionType.DELTA:
            raise ValueError(
                "action_penalty requires action type 'delta'"
            )

        if action_type == ActionType.DELTA:
            if max_action_delta <= 0:
                raise ValueError(
                    "max_action_delta must be > 0 for action type 'delta'"
                )
            self._denormalize_action = partial(
                denormalize_from_minus1_plus1,
                min_val=-float(max_action_delta),
                max_val=float(max_action_delta),
            )
        elif action_type == ActionType.ABSOLUTE:
            self._denormalize_action = partial(
                denormalize_from_minus1_plus1,
                min_val=float(control_min),
                max_val=float(control_max),
            )

        self._logger = base_logger
        self._backend = backend

        self._step_interval_tracker = CallIntervalTracker(time_multiplier=1e6)
        self._step: int = 0

        self._process_variable_max = process_variable_max
        self._reset_value = reset_value
        self._reset_steps = reset_steps
        self._control_min = control_min
        self._control_max = control_max
        self._current_control_output = BoundedValue(control_min, control_max, 0)
        self._error_prev: float = 0.0
        self._error_prev_prev: float = 0.0

        self._observe_prev_error = observe_prev_error
        self._observe_prev_prev_error = observe_prev_prev_error
        self._observe_control_output = observe_control_output

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        pv_max = float(process_variable_max)
        obs_low = [-pv_max]
        obs_high = [pv_max]
        if observe_prev_error:
            obs_low.append(-pv_max)
            obs_high.append(pv_max)
        if observe_prev_prev_error:
            obs_low.append(-pv_max)
            obs_high.append(pv_max)
        if observe_control_output:
            obs_low.append(float(control_min))
            obs_high.append(float(control_max))

        self.observation_space = gym.spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
            dtype=np.float32,
        )

    def _unpack_action_value(self, action: np.ndarray) -> float:
        return float(action[0])

    def _apply_action(self, action_value: float) -> tuple[int, bool]:
        int_value = int(round(action_value))
        if self._action_type == ActionType.DELTA:
            return self._current_control_output.add(int_value)
        else:
            self._current_control_output.value = int_value
            actual = self._current_control_output.value
            return actual, False

    def _apply_control(self, control_output: int) -> int:
        return self._backend.exchange(control_output)

    def _build_observation(
        self, process_variable: float, control_output: int
    ) -> tuple[np.ndarray, dict]:
        error = float(self._backend.setpoint - process_variable)

        components = [error]
        if self._observe_prev_error:
            components.append(self._error_prev)
        if self._observe_prev_prev_error:
            components.append(self._error_prev_prev)
        if self._observe_control_output:
            components.append(float(control_output))

        observation = np.array(components, dtype=np.float32)
        info = {
            "env.cur_error": error,
            "env.prev_error": self._error_prev,
            "env.prev_prev_error": self._error_prev_prev,
        }

        self._error_prev_prev = self._error_prev
        self._error_prev = error

        return observation, info

    def _compute_reward(
        self, observation: np.ndarray, action_norm: float
    ) -> float:
        error = float(observation[0])
        cost = abs(error)
        if self._action_penalty > 0:
            cost += self._action_penalty * abs(action_norm)
        return -cost

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        step_interval = self._step_interval_tracker.tick()
        self._step += 1

        action_norm = self._unpack_action_value(action)
        action_value = self._denormalize_action(action_norm)

        control_output, terminated = self._apply_action(action_value)
        process_variable = self._apply_control(control_output)

        observation, info = self._build_observation(process_variable, control_output)
        reward = self._compute_reward(observation, action_norm)

        self._logger.log(json.dumps({
            "source": self.LOG_SOURCE,
            "event": "step",
            "step": self._step,
            "process_variable": process_variable,
            "setpoint": self._backend.setpoint,
            "error": info["env.cur_error"],
            "error_prev": info["env.prev_error"],
            "error_prev_prev": info["env.prev_prev_error"],
            "action_norm": action_norm,
            "action_value": action_value,
            "control_output": control_output,
            "reward": reward,
            "terminated": terminated,
            "step_interval_us": step_interval,
        }))

        truncated = False
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._step = 0
        self._step_interval_tracker.reset()
        self._error_prev = 0.0
        self._error_prev_prev = 0.0

        self._backend.reset()
        self._current_control_output.value = self._reset_value

        info = {}
        for _ in range(self._reset_steps):
            process_variable = self._apply_control(self._reset_value)
            observation, info = self._build_observation(process_variable, self._reset_value)

        self._logger.log(json.dumps({
            "source": self.LOG_SOURCE,
            "event": "reset",
            "setpoint": self._backend.setpoint,
        }))

        return observation, info

    def close(self) -> None:
        self._backend.close()
        self._logger.close()

    @staticmethod
    def _create_backend(config: Config, logger: Logger) -> PlantBackend:
        backend_config = config.args.get("backend", {})
        backend_type = str(backend_config.get("type", "experimental"))

        if backend_type == "arx":
            disturbances = [
                (float(d["freq"]), float(d["amp"]))
                for d in backend_config.disturbances
            ]
            return ARXPlantBackend(
                setpoint=int(config.args.setpoint),
                a=[float(x) for x in backend_config.a],
                b=[float(x) for x in backend_config.b],
                c0=float(backend_config.c0),
                disturbances=disturbances,
                noise_std=float(backend_config.get("noise_std", 0.0)),
                dt=float(backend_config.get("dt", 0.005)),
                pv_min=float(backend_config.get("pv_min", 0.0)),
                pv_max=float(backend_config.get("pv_max", 1023.0)),
                logger=logger,
            )
        elif backend_type == "experimental":
            return ExperimentalPlantBackend(
                port=backend_config.port,
                timeout=backend_config.timeout,
                baudrate=backend_config.baudrate,
                setpoint=config.args.setpoint / 10,  # TODO: process_variable из конфига надо делить на 10
                auto_determine_setpoint=backend_config.auto_determine_setpoint,
                setpoint_determination_steps=backend_config.setpoint_determination_steps,
                setpoint_determination_max_value=backend_config.setpoint_determination_max_value,
                setpoint_determination_factor=backend_config.setpoint_determination_factor,
                control_min=config.args.control_min,
                control_max=config.args.control_max,
                log_connection=backend_config.log_connection,
                base_logger=logger,
            )
        else:
            raise ValueError(f"Unknown backend type: '{backend_type}'")

    @classmethod
    def from_config(cls, config: Config) -> "NeuralController":
        logger = AsyncFileLogger(
            log_dir=config.args.log_dir, log_file=config.args.log_file
        )
        backend = cls._create_backend(config, logger)

        action_config = config.args.action
        action_type = ActionType(str(action_config.type))
        max_action_delta = int(action_config.get("max_delta", 0))

        reward_config = config.args.get("reward", {})
        action_penalty = float(reward_config.get("action_penalty", 0.0))

        return cls(
            backend=backend,
            base_logger=logger,
            control_min=config.args.control_min,
            control_max=config.args.control_max,
            process_variable_max=config.args.process_variable_max,  # TODO: process_variable надо делить на 10
            reset_value=config.args.reset_value,
            reset_steps=config.args.reset_steps,
            observe_prev_error=bool(config.args.observe_prev_error),
            observe_prev_prev_error=bool(config.args.observe_prev_prev_error),
            observe_control_output=bool(config.args.observe_control_output),
            action_type=action_type,
            max_action_delta=max_action_delta,
            action_penalty=action_penalty,
        )
