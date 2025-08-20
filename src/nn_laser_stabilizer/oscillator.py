import random

class DuffingOscillator:
    def __init__(self, 
                 mass: float = 1.0, k_linear: float = 1.0, k_nonlinear: float = 0.1, k_damping: float = 0.1,
                 process_noise_std: float = 0.01, measurement_noise_std: float = 0.01):
        self.m = mass
        self.k_nonlinear = k_nonlinear
        self.k_linear = k_linear
        self.k_damping = k_damping

        self.process_noise_std = process_noise_std  
        self.measurement_noise_std = measurement_noise_std  

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

        self.x = self._rng.uniform(-1.0, 1.0)  
        self.v = self._rng.uniform(-1.0, 1.0)
        return self.process_variable

    def step(self, force: float, dt: float = 0.01) -> float:
        """
        Выполняем шаг по уравнению движения с шумом.
        
        x'' = (F - c * v - k_linear * x - k_nonlinear * x^3) / m + process_noise
        """
        a = (force - self.k_damping * self.v - self.k_linear * self.x - self.k_nonlinear * self.x ** 3) / self.m
        
        if self.process_noise_std > 0:
            process_noise = self._rng.gauss(0, self.process_noise_std)
            a += process_noise
        
        self.v += a * dt
        self.x += self.v * dt
        
        return self.process_variable
    
    @property
    def process_variable(self) -> float:
        measured_x = self.x
        
        if self.measurement_noise_std > 0:
            measurement_noise = self._rng.gauss(0, self.measurement_noise_std)
            measured_x += measurement_noise
            
        return measured_x
