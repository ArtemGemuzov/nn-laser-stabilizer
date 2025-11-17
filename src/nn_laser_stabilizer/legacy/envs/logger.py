import time

from torchrl.envs import EnvBase, Transform

from nn_laser_stabilizer.logging.async_file_logger import AsyncFileLogger


class PerStepLoggerAsync(Transform):
    """
    Логирует kp, ki, kd (из action) и x, setpoint (из observation) каждый шаг 
    с использованием отдельного потока записи.
    """
    def __init__(self, log_dir: str = None):
        super().__init__()
        self.logger = AsyncFileLogger(log_dir=log_dir)
        self._t = 0

    def _log_step_async(self, action_row, observation_row):
        kp, ki, kd = action_row.tolist()
        x, control_output, setpoint = observation_row.tolist()
        now = time.time()
        log_line = f"step={self._t} time={now:.6f} kp={kp:.8f} ki={ki:.8f} kd={kd:.8f} x={x:.8f} control_output={control_output:.8f} setpoint={setpoint:.8f}"
        self.logger.log(log_line)
        self._t += 1

    def _step(self, tensordict, next_tensordict):
        action = tensordict.get("action", None)
        observation = tensordict.get("observation", None)
        if action is not None and observation is not None:
            self._log_step_async(action, observation)
        return next_tensordict

    def close(self):
        self.logger.close()

    def __del__(self):
        self.close()


class LoggingEnvWrapper(EnvBase):
    """
    Логирующая обертка над окружением.
    """
    
    def __init__(self, env: EnvBase, log_dir: str = None):
        super().__init__()
        
        self.env = env
        
        self.logger = AsyncFileLogger(log_dir=log_dir)
        self._t = 0
    
    def _log_step_async(self, action_row, observation_row):
        kp, ki, kd = action_row.tolist()
        x, control_output, setpoint = observation_row.tolist()
        now = time.time()
        log_line = f"step={self._t} time={now:.6f} kp={kp:.8f} ki={ki:.8f} kd={kd:.8f} x={x:.8f} control_output={control_output:.8f} setpoint={setpoint:.8f}"
        self.logger.log(log_line)
        self._t += 1

    def _step(self, tensordict):
        next_tensordict = self.env._step(tensordict)   
     
        action = tensordict.get("action", None)
        observation = next_tensordict.get("observation", None) 
        if action is not None and observation is not None:
            self._log_step_async(action, observation)
            
        return next_tensordict

    def _reset(self, tensordict = None):
        return self.env._reset(tensordict)

    def _set_seed(self, seed: int):
        self.env._set_seed(seed)

    def set_state(self, state):
        self.env.set_state(state)

    def forward(self, tensordict):
        return self.env.forward(tensordict)

    def close(self):
        self.logger.close()
        
        if hasattr(self.env, 'close'):
            self.env.close()

    def __del__(self):
        self.close()