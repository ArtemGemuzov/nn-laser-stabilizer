from enum import Enum


class BaseEnum(Enum):
    @classmethod
    def from_str(cls, value: str):
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"Unknown {cls.__name__}: '{value}'. "
                f"Supported values: {[t.value for t in cls]}"
            )


class NetworkType(BaseEnum):
    MLP = "mlp"
    LSTM = "lstm"


class SamplerType(BaseEnum):
    SINGLE = "single"
    SEQUENCE = "sequence"

