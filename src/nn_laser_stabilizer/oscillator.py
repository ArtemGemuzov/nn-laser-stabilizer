import random

class DuffingOscillator:
    def __init__(self, mass: float = 1.0, k_linear: float = 1.0, k_nonlinear : float = 0.1 ,c_noise: float = 0.1):
        self.m = mass
        self.k_nonlinear = k_nonlinear
        self.k_linear = k_linear
        self.c_noise = c_noise

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
        Выполняем шаг по уравнению движения.
        
        x'' = (F - c * v - k_linear * x - k_nonlinear * x^3) / m
        """
        a = (force - self.c_noise * self.v - self.k_linear * self.x - self.k_nonlinear * self.x ** 3) / self.m

        self.v += a * dt
        self.x += self.v * dt
        return self.x
    
    @property
    def process_variable(self) -> float:
        return self.x