from nn_laser_stabilizer.envs.constants import DAC_MAX, ADC_MAX

def normalize_adc(value: float) -> float:
    return (value / ADC_MAX) * 2.0 - 1.0

def normalize_dac(value: float) -> float:
    return (value / DAC_MAX) * 2.0 - 1.0

def denormalize_adc(value_norm: float) -> float:
    return ((value_norm + 1.0) / 2.0) * ADC_MAX

def denormalize_dac(value_norm: float) -> float:
    return ((value_norm + 1.0) / 2.0) * DAC_MAX