class FirstOrderSystem:
    def __init__(
        self,
        time_constant: float = 0.5,
        gain: float = 1.0,
        initial_process_variable: float = 0.0,
    ) -> None:
        self.time_constant = time_constant
        self.gain = gain
        self._initial_process_variable = initial_process_variable
        self.process_variable = self._initial_process_variable
        self.reset()

    def reset(self) -> float:
        self.process_variable = self._initial_process_variable
        return self.process_variable

    def step(self, u: float, dt: float) -> float:
        """
        d(process_variable)/dt = ( -process_variable + gain * u ) / tau
        """
        dstate_dt = (-self.process_variable + self.gain * u) / self.time_constant
        self.process_variable += dstate_dt * dt
        return self.process_variable

    


