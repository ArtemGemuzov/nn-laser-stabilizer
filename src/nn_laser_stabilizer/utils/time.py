from typing import Optional
import time


class CallIntervalTracker: 
    def __init__(self, time_multiplier: float = 1.0):
        """
        Args:
            time_multiplier: Множитель для конвертации времени из секунд.
                            По умолчанию 1.0 (секунды).
                            Примеры: 1e3 (миллисекунды), 1.0 (секунды), 1e9 (наносекунды).
        """
        self._last_call_time: Optional[float] = None
        self._time_multiplier = time_multiplier
    
    def tick(self) -> float:
        """
        Отмечает текущий момент времени и возвращает интервал с предыдущего вызова.
        
        Returns:
            Интервал с предыдущего вызова (в единицах, заданных time_multiplier),
            или 0.0 если это первый вызов.
        """
        current_time = time.perf_counter()
        
        if self._last_call_time is None:
            self._last_call_time = current_time
            return 0.0
        
        interval = (current_time - self._last_call_time) * self._time_multiplier
        self._last_call_time = current_time
        return interval
    
    def reset(self) -> None:
        self._last_call_time = None
