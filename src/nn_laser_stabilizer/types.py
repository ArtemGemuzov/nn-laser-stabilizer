from enum import Enum


class NetworkType(Enum):
    MLP = "mlp"
    LSTM = "lstm"

    @classmethod
    def from_str(cls, value: str) -> "NetworkType":
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"Unknown network type: '{value}'. "
                f"Supported types: {[t.value for t in cls]}"
            )


class SamplerType(Enum):
    SINGLE = "single"
    SEQUENCE = "sequence"

    @classmethod
    def from_str(cls, value: str) -> "SamplerType":
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"Unknown sampler type: '{value}'. "
                f"Supported types: {[t.value for t in cls]}"
            )
