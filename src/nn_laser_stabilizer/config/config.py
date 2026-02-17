from typing import Any, Optional
from pathlib import Path
import yaml

from nn_laser_stabilizer.utils.paths import get_configs_dir


def _substitute_placeholders(data: Any, variables: dict[str, Any]) -> Any:
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


def _resolve_include_path(
    path_str: str,
    configs_dir: Path,
    load_variables: dict[str, str],
    visited: set[Path],
) -> dict[str, Any]:
    if not isinstance(path_str, str):
        raise ValueError(f"_include value must be a string path, got {type(path_str)}")
    resolved_path_str = path_str
    for var_name, var_value in load_variables.items():
        resolved_path_str = resolved_path_str.replace(f"{{{var_name}}}", var_value)
    include_path = Path(resolved_path_str)
    if not include_path.is_absolute():
        include_path = (configs_dir / include_path).resolve()
    if not include_path.suffix:
        include_path = include_path.with_suffix(".yaml")
    elif include_path.suffix not in (".yaml", ".yml"):
        include_path = include_path.with_suffix(".yaml")
    if not include_path.exists():
        raise FileNotFoundError(f"Included config not found: {include_path}")
    included = load_config(include_path, configs_dir=configs_dir, visited=visited)
    return included.to_dict()


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Рекурсивный merge: overrides переопределяют base, вложенные dict мержатся."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_includes(
    data: Any,
    configs_dir: Path,
    load_variables: dict[str, str],
    visited: set[Path],
) -> Any:
    if isinstance(data, dict):
        if "_include" in data:
            base = _resolve_include_path(
                data["_include"], configs_dir, load_variables, visited,
            )
            if len(data) == 1:
                return base
            # _include с дополнительными ключами: включённый конфиг
            # служит базой, остальные ключи мержатся поверх (глубокий merge).
            overrides = {
                k: _resolve_includes(v, configs_dir, load_variables, visited)
                for k, v in data.items() if k != "_include"
            }
            return _deep_merge(base, overrides)
        return {
            k: _resolve_includes(v, configs_dir, load_variables, visited)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [_resolve_includes(item, configs_dir, load_variables, visited) for item in data]
    return data


class Config:
    """
    Поддерживается только простейший YAML.
    Иммутабельный класс - после создания конфиг нельзя изменять.
    """
    __slots__ = ('_data',)
    
    def __init__(self, data: dict[str, Any] = {}):
        object.__setattr__(self, '_data', data)
    
    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"Config is immutable. Cannot set attribute '{name}'")
    
    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"Config is immutable. Cannot delete attribute '{name}'")
    
    def __getstate__(self) -> dict[str, Any]:
        return {'_data': self._data}
    
    def __setstate__(self, state: dict[str, Any]) -> None:
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
    
    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Config):
                return {k: convert(v) for k, v in value._data.items()}
            elif isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert(item) for item in value]
            elif isinstance(value, Path):
                return str(value)
            return value
        return convert(self._data)
    
    def substitute_placeholders(self, variables: dict[str, Any]) -> "Config":
        substituted_data = _substitute_placeholders(self._data, variables)
        return Config(substituted_data)

    def with_key(self, key: str, value: Any) -> "Config":
        data = self.to_dict()
        data[key] = value
        return Config(data)

    def save(self, path: Path) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def find_config_path(relative_config_path: str | Path) -> Path:
    rel_path = Path(relative_config_path)
    if rel_path.is_absolute():
        return rel_path
    
    return get_configs_dir() / rel_path


def load_config(config_path: Path, configs_dir: Optional[Path] = None, visited: Optional[set[Path]] = None) -> Config:
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
        
        load_variables = {"CONFIGS_DIR": str(configs_dir)}
        config_data = _resolve_includes(config_data, configs_dir, load_variables, visited)
        
        return Config(config_data)
    finally:
        visited.remove(config_path)


def find_and_load_config(relative_config_path: str | Path) -> Config:
    """
    Находит и загружает конфиг по относительному пути.
    
    Args:
        relative_config_path: Относительный путь к конфигу (например, "train_from_csv" 
                             или "neural_controller.yaml"). Может быть без расширения.
    
    Returns:
        Config: Загруженный объект конфигурации.
    
    Raises:
        FileNotFoundError: Если конфиг не найден.
    """
    config_path = find_config_path(relative_config_path)
    return load_config(config_path)

