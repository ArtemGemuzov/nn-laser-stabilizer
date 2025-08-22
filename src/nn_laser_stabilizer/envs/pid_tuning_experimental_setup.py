from typing import Protocol, Tuple, Optional


class PidTuningExperimentalSetup(Protocol):
    def step(self, kp: float, ki: float, kd: float) -> Tuple[float, float, float]:
        """
        Выполняет шаг эксперимента: применяет новые параметры PID и возвращает
        текущее состояние системы.

        Returns:
            Tuple[process_variable, control_output, setpoint]
        """
        pass
    
    def reset(self) -> Tuple[float, float, float]:
        """
        Сбрасывает экспериментальную установку в начальное состояние.

        Returns:
            Tuple[process_variable, control_output, setpoint]
        """
        pass

    def set_seed(self, seed: Optional[int]) -> None:
        """
        Устанавливает зерно генератора случайных.
        """
        pass