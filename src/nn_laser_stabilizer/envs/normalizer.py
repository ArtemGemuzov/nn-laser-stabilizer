class Normalizer: 
    def __init__(self, 
                 process_variable_max: float,
                 control_output_max: float,
                 kp_min: float,
                 kp_max: float,
                 ki_min: float,
                 ki_max: float,
                 kd_min: float,
                 kd_max: float,
                 default_kp: float,
                 default_ki: float,
                 default_kd: float):
        self.process_variable_max = process_variable_max
        self.control_output_max = control_output_max
        self.kp_min = kp_min
        self.kp_max = kp_max
        self.ki_min = ki_min
        self.ki_max = ki_max
        self.kd_min = kd_min
        self.kd_max = kd_max
        self.default_kp = default_kp
        self.default_ki = default_ki
        self.default_kd = default_kd
    
    def normalize_process_variable(self, value: float) -> float:
        return self._normalize(value, 0.0, self.process_variable_max)
    
    def normalize_control_output(self, value: float) -> float:
        return self._normalize(value, 0.0, self.control_output_max)
    
    def denormalize_process_variable(self, value_norm: float) -> float:
        return self._denormalize(value_norm, 0.0, self.process_variable_max)
    
    def denormalize_control_output(self, value_norm: float) -> float:
        return self._denormalize(value_norm, 0.0, self.control_output_max)

    def normalize_kp(self, value: float) -> float:
        return self._normalize(value, self.kp_min, self.kp_max)
    
    def normalize_ki(self, value: float) -> float:
        return self._normalize(value, self.ki_min, self.ki_max)
    
    def normalize_kd(self, value: float) -> float:
        return self._normalize(value, self.kd_min, self.kd_max)
    
    def denormalize_kp(self, value_norm: float) -> float:
        return self._denormalize(value_norm, self.kp_min, self.kp_max)
    
    def denormalize_ki(self, value_norm: float) -> float:
        return self._denormalize(value_norm, self.ki_min, self.ki_max)
    
    def denormalize_kd(self, value_norm: float) -> float:
        return self._denormalize(value_norm, self.kd_min, self.kd_max)
    
    def standardize(self, value: float, mean: float, std: float) -> float:
        if std == 0.0:
            return value
        return (value - mean) / (std + 10E-8)
    
    def destandardize(self, value_std: float, mean: float, std: float) -> float:
        return value_std * std + mean
    
    def _normalize(self, value: float, min_val: float, max_val: float, 
                       target_min: float = -1.0, target_max: float = 1.0) -> float:
        if max_val == min_val:
            return (target_min + target_max) / 2.0
        return (value - min_val) / (max_val - min_val) * (target_max - target_min) + target_min
    
    def _denormalize(self, value_norm: float, min_val: float, max_val: float,
                         source_min: float = -1.0, source_max: float = 1.0) -> float:
        if source_max == source_min:
            return (min_val + max_val) / 2.0
        return (value_norm - source_min) / (source_max - source_min) * (max_val - min_val) + min_val


def make_normalizer(config) -> Normalizer:
    env_config = config.env
    return Normalizer(
        process_variable_max=float(env_config.process_variable_max),
        control_output_max=float(env_config.control_output_max),
        kp_min=float(env_config.kp_min),
        kp_max=float(env_config.kp_max),
        ki_min=float(env_config.ki_min),
        ki_max=float(env_config.ki_max),
        kd_min=float(env_config.kd_min),
        kd_max=float(env_config.kd_max),
        default_kp=float(env_config.default_kp),
        default_ki=float(env_config.default_ki),
        default_kd=float(env_config.default_kd)
    )
