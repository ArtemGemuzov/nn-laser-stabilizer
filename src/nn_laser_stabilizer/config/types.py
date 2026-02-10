from nn_laser_stabilizer.enum import BaseEnum


class NetworkType(BaseEnum):
    MLP = "mlp"
    LSTM = "lstm"


class SamplerType(BaseEnum):
    SINGLE = "single"
    SEQUENCE = "sequence"


class ExplorationType(BaseEnum):
    NONE = "none"
    RANDOM = "random"
    NOISY = "noisy"
    OU = "ou"


class AlgorithmType(BaseEnum):
    TD3 = "td3"
    TD3BC = "td3bc"
    BC = "bc"
    SAC = "sac"

