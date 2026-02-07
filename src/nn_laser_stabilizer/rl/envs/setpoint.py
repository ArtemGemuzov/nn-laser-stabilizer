from typing import Callable


def determine_setpoint(
    send_control_and_get_pv: Callable[[int], int],
    steps: int,
    max_value: int,
    factor: float,
) -> tuple[int, int, int]:
    """
    Сканирует диапазон [0, max_value] с заданным числом шагов, собирает min/max
    process variable и возвращает setpoint = min_pv + factor * (max_pv - min_pv).

    send_control_and_get_pv: функция (control_value: int) -> process_variable: int.
    """
    if steps <= 1:
        raise ValueError(f"steps ({steps}) must be greater than 1")

    min_pv = float("inf")
    max_pv = float("-inf")

    for step in range(steps):
        progress = step / (steps - 1)
        control_value = int(progress * max_value)
        process_variable = send_control_and_get_pv(control_value)
        min_pv = min(min_pv, process_variable)
        max_pv = max(max_pv, process_variable)

    min_pv_int = int(min_pv)
    max_pv_int = int(max_pv)
    setpoint = int(round(min_pv + factor * (max_pv - min_pv)))
    return setpoint, min_pv_int, max_pv_int
