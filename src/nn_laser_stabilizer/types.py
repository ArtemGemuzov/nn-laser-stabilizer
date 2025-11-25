from enum import Enum


class NetworkType(Enum):
    MLP = "mlp"
    LSTM = "lstm"


class SamplerType(Enum):
    SINGLE = "single"
    SEQUENCE = "sequence"

