from pathlib import Path
import argparse

from nn_laser_stabilizer.connection import create_connection
from nn_laser_stabilizer.pid import ConnectionToPid
from nn_laser_stabilizer.plant import determine_setpoint
from nn_laser_stabilizer.experiment.config import load_config, find_config_path


def main(
    port: str,
    timeout: float,
    baudrate: int,
    steps: int,
    max_value: int,
    factor: float,
) -> None:
    connection = create_connection(
        port=port,
        timeout=timeout,
        baudrate=baudrate,
    )
    pid_connection = ConnectionToPid(connection=connection)
    
    try:
        pid_connection.open()
        
        print(f"Определение setpoint: steps={steps}, max_value={max_value}")
        print("Выполняется сканирование...")
        
        setpoint, min_pv, max_pv = determine_setpoint(
            pid_connection=pid_connection,
            steps=steps,
            max_value=max_value,
            factor=factor,
        )
        
        print(f"\nРезультат:")
        print(f"  Setpoint: {setpoint}")
        print(f"  Min PV: {min_pv}")
        print(f"  Max PV: {max_pv}")
        
    finally:
        pid_connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Определение setpoint путем линейного сканирования control_min и control_max."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pid_delta_tuning",
        help="Относительный путь к конфигу внутри 'configs/' (без .yaml). По умолчанию: pid_delta_tuning",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    config_path = find_config_path(Path(args.config))
    config = load_config(config_path)
    
    env_args = config.env.args
    
    port = env_args.port
    timeout = env_args.timeout
    baudrate = env_args.baudrate
    steps = env_args.setpoint_determination_steps
    max_value = env_args.setpoint_determination_max_value
    factor = env_args.setpoint_determination_factor
    
    main(
        port=port,
        timeout=timeout,
        baudrate=baudrate,
        steps=steps,
        max_value=max_value,
        factor=factor,
    )
