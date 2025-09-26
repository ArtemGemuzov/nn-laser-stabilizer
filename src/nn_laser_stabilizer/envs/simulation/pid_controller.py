class PIDController:
    def __init__(self, setpoint: float = 1.0):
        self.setpoint = setpoint

        self.kp = 0.0
        self.ki = 0.0
        self.kd = 0.0

        self.integral = 0.0
        self.prev_error = 0.0

    def set_params(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def __call__(self, process_variable: float, dt : float = 0.01) -> float:
        error = self.setpoint - process_variable
        
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        return control