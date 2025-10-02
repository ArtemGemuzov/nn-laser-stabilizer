from dataclasses import dataclass


@dataclass
class ControlLimitConfig:
    default_min: float
    default_max: float
    force_min_value: float
    force_max_value: float
    lower_force_condition_threshold: float
    upper_force_condition_threshold: float
    enforcement_steps: int


class ControlLimitManager:
    """
    Управляет текущими пределами управления и правилом принудительного минимума.
    """

    def __init__(self, config: ControlLimitConfig):
        self.config = config
        self._current_min = config.default_min
        self._current_max = config.default_max
        self._force_steps_left = 0

        self.reset()

    def reset(self) -> None:
        self._current_min = self.config.default_min
        self._current_max = self.config.default_max
        self._force_steps_left = 0

    def apply_rule(self, control_output: float) -> None:
        if (control_output < self.config.lower_force_condition_threshold or 
            control_output > self.config.upper_force_condition_threshold):
            self._force_steps_left = self.config.enforcement_steps
        elif self._force_steps_left > 0:
            self._force_steps_left -= 1

    def get_limits_for_step(self) -> tuple[float, float]:
        if self._force_steps_left > 0:
            return self.config.force_min_value, self.config.force_max_value
        return self._current_min, self._current_max



def make_control_limit_manager(
    *,
    default_min: float,
    default_max: float,
    force_min_value: float,
    force_max_value: float,
    lower_force_condition_threshold: float,
    upper_force_condition_threshold: float,
    enforcement_steps: int,
) -> ControlLimitManager:
    """Фабрика для создания ControlLimitManager из простых аргументов."""
    config = ControlLimitConfig(
        default_min=default_min,
        default_max=default_max,
        force_min_value=force_min_value,
        force_max_value=force_max_value,
        lower_force_condition_threshold=lower_force_condition_threshold,
        upper_force_condition_threshold=upper_force_condition_threshold,
        enforcement_steps=enforcement_steps,
    )
    return ControlLimitManager(config)
