class DampedOscillator:
    def __init__(self, dt: float = 0.01, mass: float = 1.0, k: float = 1.0, c: float = 0.1):
        """
        m * x'' + c * x' + k * x = F

        Args:
            dt: шаг по времени
            mass: масса
            k: коэффициент жесткости (пружины)
            c: коэффициент демпфирования
        """
        self.dt = dt
        self.m = mass
        self.k = k
        self.c = c

        # состояние: x (смещение), v (скорость)
        self.x = 0.0
        self.v = 0.0

    def reset(self) -> float:
        self.x = 0.0
        self.v = 0.0
        return self.x

    def step(self, force: float) -> float:
        """Выполняем шаг по уравнениям движения.
        
            x'' = (F - c*v - k*x) / m
        """
        a = (force - self.c * self.v - self.k * self.x) / self.m

        self.v += a * self.dt
        self.x += self.v * self.dt
        return self.x