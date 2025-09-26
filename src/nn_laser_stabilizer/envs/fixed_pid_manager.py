from typing import Optional, Tuple


class FixedPidManager:
    """
    Управляет фиксированными коэффициентами PID регулятора.
    
    Если все три коэффициента (kp, ki, kd) заданы, то они будут использоваться
    вместо значений, поступающих от агента.
    """
    
    def __init__(self, 
                 fixed_kp: Optional[float] = None,
                 fixed_ki: Optional[float] = None, 
                 fixed_kd: Optional[float] = None):
        """
        Инициализирует менеджер фиксированных коэффициентов.
        
        Args:
            fixed_kp: Фиксированное значение Kp
            fixed_ki: Фиксированное значение Ki  
            fixed_kd: Фиксированное значение Kd
        """
        if all(x is not None for x in [fixed_kp, fixed_ki, fixed_kd]):
            self._fixed_coefficients = (float(fixed_kp), float(fixed_ki), float(fixed_kd))
        else:
            self._fixed_coefficients = None
    
    def get_coefficients(self, agent_kp: float, agent_ki: float, agent_kd: float) -> Tuple[float, float, float]:
        """
        Возвращает коэффициенты для использования.
        
        Если заданы фиксированные коэффициенты, возвращает их.
        Иначе возвращает коэффициенты от агента.
        
        Args:
            agent_kp: Коэффициент Kp от агента
            agent_ki: Коэффициент Ki от агента
            agent_kd: Коэффициент Kd от агента
            
        Returns:
            Кортеж (kp, ki, kd) для использования
        """
        if self._fixed_coefficients is not None:
            return self._fixed_coefficients
        return (agent_kp, agent_ki, agent_kd)
