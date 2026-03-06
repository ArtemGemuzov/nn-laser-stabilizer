from typing import Optional

import numpy as np
import gymnasium as gym

from nn_laser_stabilizer.utils.enum import BaseEnum
from nn_laser_stabilizer.utils.logger import Logger, NoOpLogger
from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.envs.base_env import BaseEnv
from nn_laser_stabilizer.utils.bounded_value import BoundedValue
from nn_laser_stabilizer.rl.envs.arx_plant_backend import ARXPlantBackend
from nn_laser_stabilizer.rl.envs.plant_backend import ExperimentalPlantBackend, PlantBackend
from nn_laser_stabilizer.utils.normalize import (
    denormalize_from_minus1_plus1,
    normalize_to_minus1_plus1,
)


class BackendType(BaseEnum):
    EXPERIMENTAL = "experimental"
    ARX = "arx"


class ActionType(BaseEnum):
    DELTA = "delta"
    ABSOLUTE = "absolute"


class RewardType(BaseEnum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"


class ErrorTermFn:
    def compute(self, error: float) -> float:
        raise NotImplementedError


class LinearErrorTermFn(ErrorTermFn):
    def __init__(self, *, denominator: float) -> None:
        if denominator <= 0:
            raise ValueError("reward.params.linear_denominator must be > 0")
        self._inv_denominator = 1.0 / denominator

    def compute(self, error: float) -> float:
        return -self._inv_denominator * abs(error)


class QuadraticErrorTermFn(ErrorTermFn):
    def __init__(self, *, denominator: float) -> None:
        if denominator <= 0:
            raise ValueError("reward.params.quadratic_denominator must be > 0")
        self._inv_denominator = 1.0 / denominator

    def compute(self, error: float) -> float:
        normalized_abs_error = self._inv_denominator * abs(error)
        return -(normalized_abs_error ** 2)


class ExponentialErrorTermFn(ErrorTermFn):
    def __init__(self, *, sigma: float) -> None:
        if sigma <= 0:
            raise ValueError("reward.params.sigma must be > 0")
        self._inv_sigma = 1.0 / sigma

    def compute(self, error: float) -> float:
        return float(np.exp(-self._inv_sigma * abs(error)))


class GaussianErrorTermFn(ErrorTermFn):
    def __init__(self, *, sigma: float) -> None:
        if sigma <= 0:
            raise ValueError("reward.params.sigma must be > 0")
        sigma2x2 = 2.0 * (sigma ** 2)
        self._inv_sigma2x2 = 1.0 / sigma2x2

    def compute(self, error: float) -> float:
        abs_error = abs(error)
        return float(np.exp(-(abs_error ** 2) * self._inv_sigma2x2))


class NeuralController(BaseEnv):
    def __init__(
        self,
        *,
        backend: PlantBackend,
        control_min: int,
        control_max: int,
        process_variable_min: int,
        process_variable_max: int,
        reset_value: int,
        reset_steps: int,
        observe_prev_error: bool,
        observe_prev_prev_error: bool,
        observe_control_output: bool,
        action_type: ActionType,
        max_action_delta: int = 0,
        action_penalty: float = 0.0,
        control_penalty: float = 0.0,
        barrier_penalty: float = 0.0,
        terminal_penalty: float = 0.0,
        alive_bonus: float = 0.0,
        reward_type: RewardType = RewardType.QUADRATIC,
        reward_params: Optional[Config | dict[str, float]] = None,
        normalize_obs: bool = False,
        error_normalixation_factor: float = 60.0,
    ):
        super().__init__()

        self._action_type = action_type
        self._action_penalty = action_penalty
        self._control_penalty = control_penalty
        self._barrier_penalty = barrier_penalty
        self._terminal_penalty = terminal_penalty
        self._alive_bonus = alive_bonus
        reward_params_cfg = (
            reward_params
            if isinstance(reward_params, Config)
            else Config(reward_params or {})
        )
        self._error_term_fn = self._create_error_term_fn(
            reward_type=reward_type,
            reward_params=reward_params_cfg,
        )
        self._control_midpoint = (control_min + control_max) / 2.0
        self._control_half_range = (control_max - control_min) / 2.0

        if action_penalty > 0 and action_type != ActionType.DELTA:
            raise ValueError(
                "action_penalty requires action type 'delta'"
            )

        if action_type == ActionType.DELTA:
            if max_action_delta <= 0:
                raise ValueError(
                    "max_action_delta must be > 0 for action type 'delta'"
                )
            self._denormalize_action = lambda value: denormalize_from_minus1_plus1(
                value,
                min_val=-float(max_action_delta),
                max_val=float(max_action_delta),
            )
        elif action_type == ActionType.ABSOLUTE:
            self._denormalize_action = lambda value: denormalize_from_minus1_plus1(
                value,
                min_val=float(control_min),
                max_val=float(control_max),
            )

        self._backend = backend

        self._process_variable_min = process_variable_min
        self._process_variable_max = process_variable_max
        self._pv_range = float(process_variable_max - process_variable_min)
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
        self._normalize_obs = normalize_obs
        if error_normalixation_factor <= 0:
            raise ValueError("error_normalixation_factor must be > 0")
        self._error_normalixation_factor = float(error_normalixation_factor)

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        obs_dim = 1
        if observe_prev_error:
            obs_dim += 1
        if observe_prev_prev_error:
            obs_dim += 1
        if observe_control_output:
            obs_dim += 1

        if normalize_obs:
            self.observation_space = gym.spaces.Box(
                low=np.full(obs_dim, -1.0, dtype=np.float32),
                high=np.full(obs_dim, 1.0, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            error_bound = self._pv_range
            obs_low = [-error_bound]
            obs_high = [error_bound]
            if observe_prev_error:
                obs_low.append(-error_bound)
                obs_high.append(error_bound)
            if observe_prev_prev_error:
                obs_low.append(-error_bound)
                obs_high.append(error_bound)
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
            "error": error,
            "prev_error": self._error_prev,
            "prev_prev_error": self._error_prev_prev,
        }

        self._error_prev_prev = self._error_prev
        self._error_prev = error

        return observation, info

    def _normalize_error(self, value: float) -> float:
        return float(np.clip(value / self._error_normalixation_factor, -1.0, 1.0))

    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        idx = 0
        observation[idx] = self._normalize_error(float(observation[idx]))
        idx += 1
        if self._observe_prev_error:
            observation[idx] = self._normalize_error(float(observation[idx]))
            idx += 1
        if self._observe_prev_prev_error:
            observation[idx] = self._normalize_error(float(observation[idx]))
            idx += 1
        if self._observe_control_output:
            observation[idx] = normalize_to_minus1_plus1(
                float(observation[idx]),
                min_val=float(self._control_min),
                max_val=float(self._control_max),
            )
        return observation

    def _compute_reward(
        self,
        error: float,
        action_norm: float,
        control_output: int,
        terminated: bool,
    ) -> dict[str, float]:
        error_term = self._error_term_fn.compute(error)

        action_cost = 0.0
        if self._action_penalty > 0:
            action_cost = self._action_penalty * abs(action_norm)

        control_cost = 0.0
        if self._control_penalty > 0:
            deviation = abs(control_output - self._control_midpoint) / self._control_half_range
            control_cost = self._control_penalty * deviation

        barrier_cost = 0.0
        if self._barrier_penalty != 0.0:
            u_norm = normalize_to_minus1_plus1(
                float(control_output),
                min_val=float(self._control_min),
                max_val=float(self._control_max),
            )
            barrier_cost = self._barrier_penalty * float(-np.log(1.0 - abs(u_norm) + 1e-6))

        terminal_cost = self._terminal_penalty if terminated else 0.0
        alive_reward = self._alive_bonus if not terminated else 0.0

        total_cost = action_cost + control_cost + barrier_cost + terminal_cost
        reward = error_term - total_cost + alive_reward
        return {
            "reward_error_term": error_term,
            "cost_action": action_cost,
            "cost_control": control_cost,
            "cost_barrier": barrier_cost,
            "cost_terminal": terminal_cost,
            "reward_alive": alive_reward,
            "reward": reward,
        }

    @staticmethod
    def _create_error_term_fn(
        *,
        reward_type: RewardType,
        reward_params: Config,
    ) -> ErrorTermFn:
        if reward_type == RewardType.LINEAR:
            return LinearErrorTermFn(
                denominator=float(reward_params.linear_denominator),
            )
        if reward_type == RewardType.QUADRATIC:
            return QuadraticErrorTermFn(
                denominator=float(reward_params.quadratic_denominator),
            )
        if reward_type == RewardType.EXPONENTIAL:
            return ExponentialErrorTermFn(
                sigma=float(reward_params.sigma),
            )
        if reward_type == RewardType.GAUSSIAN:
            return GaussianErrorTermFn(
                sigma=float(reward_params.sigma),
            )
        raise ValueError(f"Unknown reward type: '{reward_type.value}'")

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_norm = self._unpack_action_value(action)
        action_value = self._denormalize_action(action_norm)

        control_output, terminated = self._apply_action(action_value)
        process_variable = self._apply_control(control_output)

        observation, info = self._build_observation(process_variable, control_output)

        if self._normalize_obs:
            observation = self._normalize_observation(observation)

        reward_info = self._compute_reward(
            error=info["error"],
            action_norm=action_norm,
            control_output=control_output,
            terminated=terminated,
        )

        truncated = False

        info.update(reward_info)
        info.update({
            "process_variable": process_variable,
            "setpoint": self._backend.setpoint,
            "action_norm": action_norm,
            "action_value": action_value,
            "control_output": control_output,
            "terminated": terminated,
            "truncated": truncated
        })

        return observation, reward_info["reward"], terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._error_prev = 0.0
        self._error_prev_prev = 0.0

        self._backend.reset()
        self._current_control_output.value = self._reset_value

        info = {}
        for _ in range(self._reset_steps):
            process_variable = self._apply_control(self._reset_value)
            observation, info = self._build_observation(process_variable, self._reset_value)

        if self._normalize_obs:
            observation = self._normalize_observation(observation)

        info["setpoint"] = self._backend.setpoint

        return observation, info

    def close(self) -> None:
        self._backend.close()

    @staticmethod
    def _create_backend(config: Config, logger: Logger) -> PlantBackend:
        backend_config = config.args.get("backend", {})
        backend_type = BackendType.from_str(str(backend_config.type))

        if backend_type == BackendType.ARX:
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
                setpoint_override_probability=float(
                    backend_config.get("setpoint_override_probability", 0.0)
                ),
                pv_min=float(backend_config.get("pv_min", 0.0)),
                pv_max=float(backend_config.get("pv_max", 1023.0)),
                logger=logger,
            )
        elif backend_type == BackendType.EXPERIMENTAL:
            return ExperimentalPlantBackend(
                port=backend_config.port,
                timeout=backend_config.timeout,
                baudrate=backend_config.baudrate,
                setpoint=config.args.setpoint / 10,
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
            raise ValueError(f"Unknown backend type: '{backend_type.value}'")

    @classmethod
    def from_config(
        cls, config: Config, logger: Logger | None = None
    ) -> "NeuralController":
        if logger is None:
            logger = NoOpLogger()
        backend = cls._create_backend(config, logger)

        action_config = config.args.action
        action_type = ActionType.from_str(str(action_config.type))
        max_action_delta = int(action_config.get("max_delta", 0))

        reward_config = config.args.get("reward", {})
        reward_type = RewardType.from_str(reward_config.type)
        reward_params = reward_config.params
        action_penalty = float(reward_config.get("action_penalty", 0.0))
        control_penalty = float(reward_config.get("control_penalty", 0.0))
        barrier_penalty = float(reward_config.get("barrier_penalty", 0.0))
        terminal_penalty = float(reward_config.get("terminal_penalty", 0.0))
        alive_bonus = float(reward_config.get("alive_bonus", 0.0))

        return cls(
            backend=backend,
            control_min=config.args.control_min,
            control_max=config.args.control_max,
            process_variable_min=int(config.args.process_variable_min) // 10, # TODO: переменные, относящиеся к PV надр делить на 10
            process_variable_max=int(config.args.process_variable_max) // 10,
            reset_value=config.args.reset_value,
            reset_steps=config.args.reset_steps,
            observe_prev_error=bool(config.args.observe_prev_error),
            observe_prev_prev_error=bool(config.args.observe_prev_prev_error),
            observe_control_output=bool(config.args.observe_control_output),
            action_type=action_type,
            max_action_delta=max_action_delta,
            action_penalty=action_penalty,
            control_penalty=control_penalty,
            barrier_penalty=barrier_penalty,
            terminal_penalty=terminal_penalty,
            alive_bonus=alive_bonus,
            reward_type=reward_type,
            reward_params=reward_params,
            normalize_obs=bool(config.args.get("normalize_obs", False)),
            error_normalixation_factor=float(
                config.args.get("error_normalixation_factor", 60.0)
            ),
        )
