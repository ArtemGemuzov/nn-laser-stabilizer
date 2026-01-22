from nn_laser_stabilizer.enum_base import BaseEnum


class NetworkType(BaseEnum):
    MLP = "mlp"
    LSTM = "lstm"


class SamplerType(BaseEnum):
    SINGLE = "single"
    SEQUENCE = "sequence"


class ExplorationType(BaseEnum):
    NONE = "none"
    RANDOM = "random"

