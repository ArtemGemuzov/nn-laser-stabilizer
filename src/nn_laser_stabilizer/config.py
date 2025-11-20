from pathlib import Path
from typing import Any, Dict
import yaml


class Config:
    """
    Поддерживается только простейший YAML.
    Иммутабельный класс - после создания конфиг нельзя изменять.
    """
    __slots__ = ('_data',)
    
    def __init__(self, data: Dict[str, Any]):
        object.__setattr__(self, '_data', data)
    
    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"Config is immutable. Cannot set attribute '{name}'")
    
    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"Config is immutable. Cannot delete attribute '{name}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        cur_value = self._data
        
        for cur_key in keys:
            if not isinstance(cur_value, dict) or cur_key not in cur_value:
                return default
            cur_value = cur_value[cur_key]
        
        return cur_value
    
    def __getattr__(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise AttributeError(f"Config has no attribute '{key}'")
        if isinstance(value, dict):
            return Config(value)
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        def convert(value):
            if isinstance(value, Config):
                return {k: convert(v) for k, v in value._data.items()}
            elif isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert(item) for item in value]
            return value
        return convert(self._data)


def load_config(config_path: Path) -> Config:
    config_path = Path(config_path).resolve()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    if config_data is None:
        config_data = {}
    
    return Config(config_data)

