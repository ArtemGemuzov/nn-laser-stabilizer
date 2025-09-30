from nn_laser_stabilizer.envs.constants import DAC_MAX, ADC_MAX, KP_MIN, KP_MAX, KI_MIN, KI_MAX, KD_MIN, KD_MAX

def normalize_adc(value: float) -> float:
    return (value / ADC_MAX) * 2.0 - 1.0

def normalize_dac(value: float) -> float:
    return (value / DAC_MAX) * 2.0 - 1.0

def denormalize_adc(value_norm: float) -> float:
    return ((value_norm + 1.0) / 2.0) * ADC_MAX

def denormalize_dac(value_norm: float) -> float:
    return ((value_norm + 1.0) / 2.0) * DAC_MAX

def standardize(value: float, mean: float, std: float) -> float:
    """
    Стандартизация: (value - mean) / std
    """
    if std == 0.0:
        return value
    return (value - mean) / std

def destandardize(value_std: float, mean: float, std: float) -> float:
    """
    Обратная стандартизация: value_std * std + mean
    """
    return value_std * std + mean

def normalize_kp(value: float) -> float:
    """
    Нормализация Kp параметра в диапазон [-1, 1]
    """
    return (value - KP_MIN) / (KP_MAX - KP_MIN) * 2.0 - 1.0

def normalize_ki(value: float) -> float:
    """
    Нормализация Ki параметра в диапазон [-1, 1]
    """
    return (value - KI_MIN) / (KI_MAX - KI_MIN) * 2.0 - 1.0

def normalize_kd(value: float) -> float:
    """
    Нормализация Kd параметра в диапазон [-1, 1]
    """
    return (value - KD_MIN) / (KD_MAX - KD_MIN) * 2.0 - 1.0

def denormalize_kp(value_norm: float) -> float:
    """
    Денормализация Kp параметра из диапазона [-1, 1] в исходный диапазон
    """
    return ((value_norm + 1.0) / 2.0) * (KP_MAX - KP_MIN) + KP_MIN

def denormalize_ki(value_norm: float) -> float:
    """
    Денормализация Ki параметра из диапазона [-1, 1] в исходный диапазон
    """
    return ((value_norm + 1.0) / 2.0) * (KI_MAX - KI_MIN) + KI_MIN

def denormalize_kd(value_norm: float) -> float:
    """
    Денормализация Kd параметра из диапазона [-1, 1] в исходный диапазон
    """
    return ((value_norm + 1.0) / 2.0) * (KD_MAX - KD_MIN) + KD_MIN