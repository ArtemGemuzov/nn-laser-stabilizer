from enum import Enum


class BaseEnum(Enum):
    """Базовый класс для кастомных enum."""
    
    @classmethod
    def from_str(cls, value: str):
        """
        Создает экземпляр enum из строки.
        
        Args:
            value: Строковое значение enum
        
        Returns:
            Экземпляр enum
        
        Raises:
            ValueError: Если значение не поддерживается
        """
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"Unknown {cls.__name__}: '{value}'. "
                f"Supported values: {[t.value for t in cls]}"
            )

