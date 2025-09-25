from dataclasses import dataclass


@dataclass
class ControlLimitConfig:
    default_min: float
    default_max: float
    force_min_value: float
    force_condition_threshold: float
    enforcement_steps: int


class ControlLimitManager:
    """
    Управляет текущими пределами управления и правилом принудительного минимума.
    """

    def __init__(self, config: ControlLimitConfig):
        self.config = config
        self._current_min = float(config.default_min)
        self._current_max = float(config.default_max)
        self._force_steps_left = 0

        self.reset()

    def reset(self) -> None:
        self._current_min = float(self.config.default_min)
        self._current_max = float(self.config.default_max)
        self._force_steps_left = 0

    def apply_rule(self, control_output: float) -> None:
        if control_output < float(self.config.force_condition_threshold):
            self._force_steps_left = int(self.config.enforcement_steps)
        elif self._force_steps_left > 0:
            self._force_steps_left -= 1

    def get_limits_for_step(self) -> tuple[float, float]:
        if self._force_steps_left > 0:
            return float(self.config.force_min_value), self._current_max
        return self._current_min, self._current_max


