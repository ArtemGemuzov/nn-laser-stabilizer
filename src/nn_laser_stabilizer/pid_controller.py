class PIDController:
    def __init__(self, setpoint: float = 1.0, dt: float = 0.01):
        self.setpoint = setpoint
        self.dt = dt

        # параметры
        self.kp = 1.0
        self.ki = 0.0
        self.kd = 0.0

        # внутреннее состояние
        self.integral = 0.0
        self.prev_error = 0.0

    def set_params(self, kp: float, ki: float, kd: float):
        self.kp, self.ki, self.kd = kp, ki, kd

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, process_variable: float) -> float:
        error = self.setpoint - process_variable
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        return control