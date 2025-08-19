import random

class Oscillator:
    def __init__(self, mass: float = 1.0, k: float = 1.0, c: float = 0.1):
        """
        m * x'' + c * x' + k * x = F

        Args:
            mass: масса
            k: коэффициент жесткости
            c: коэффициент сопротивления
        """
        self.m = mass
        self.k = k
        self.c = c

        self._rng = None

        self.x = 0.0
        self.v = 0.0
        self.reset()

    def set_seed(self, seed: int) -> None:
        """Задает seed для генератора случайных чисел."""
        self._rng = random.Random(seed)

    def reset(self) -> float:
        """Сбрасывает состояние осциллятора, задавая случайные начальные значения."""
        if self._rng is None:
            self._rng = random.Random()

        self.x = self._rng.uniform(-10.0, 10.0)  
        self.v = self._rng.uniform(-5.0, 5.0)
        return self.x

    def step(self, force: float, dt : float = 0.01) -> float:
        """
        Выполняем шаг по уравнениям движения.
        
        x'' = (F - c*v - k*x) / m
        """
        a = (force - self.c * self.v - self.k * self.x) / self.m

        self.v += a * dt
        self.x += self.v * dt
        return self.x
    
    @property
    def process_variable(self) -> float:
        return self.x