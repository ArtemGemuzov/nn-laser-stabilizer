from typing import Any, Dict, Set
from pathlib import Path
import yaml


CONFIGS_DIR = Path("configs")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    
    for key, value in override.items():
        if key == '_extends':
            continue
        
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def _substitute_placeholders(data: Any, variables: Dict[str, Any]) -> Any:
    if isinstance(data, str):
        result = data
        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(var_value))
        return result
    elif isinstance(data, dict):
        return {k: _substitute_placeholders(v, variables) for k, v in data.items()}
    elif isinstance(data, list):
        return [_substitute_placeholders(item, variables) for item in data]
    else:
        return data


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
    
    def __getstate__(self) -> Dict[str, Any]:
        return {'_data': self._data}
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        object.__setattr__(self, '_data', state['_data'])
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        cur_value = self._data
        
        for cur_key in keys:
            if not isinstance(cur_value, dict) or cur_key not in cur_value:
                return default
            cur_value = cur_value[cur_key]
        
        if isinstance(cur_value, dict):
            return Config(cur_value)
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
    
    def substitute_placeholders(self, variables: Dict[str, Any]) -> "Config":
        substituted_data = _substitute_placeholders(self._data, variables)
        return Config(substituted_data)
    
    def save(self, path: Path) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def find_config_path(relative_config_path: str | Path) -> Path:
    rel_path = Path(relative_config_path)
    if rel_path.is_absolute():
        return rel_path
    
    return CONFIGS_DIR / rel_path


def load_config(config_path: Path, configs_dir: Path = None, visited: Set[Path] = None) -> Config:
    config_path = Path(config_path)
    
    if not config_path.suffix:
        config_path = config_path.with_suffix('.yaml')
    elif config_path.suffix not in ('.yaml', '.yml'):
        config_path = config_path.with_suffix('.yaml')
    
    config_path = config_path.resolve()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if configs_dir is None:
        configs_dir = config_path.parent
    
    if visited is None:
        visited = set()
    
    if config_path in visited:
        raise ValueError(f"Circular dependency detected: {config_path} is already being loaded")
    
    visited.add(config_path)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        if config_data is None:
            config_data = {}
        
        extends_path = config_data.get('_extends')
        
        if extends_path:
            if isinstance(extends_path, str):
                if not Path(extends_path).is_absolute():
                    base_config_path = configs_dir / extends_path
                else:
                    base_config_path = Path(extends_path)
                
                base_config = load_config(base_config_path, configs_dir=configs_dir, visited=visited)
                base_data = base_config.to_dict()
                
                config_data = _deep_merge(base_data, config_data)
            
            config_data.pop('_extends', None)
        
        return Config(config_data)
    finally:
        visited.remove(config_path)

