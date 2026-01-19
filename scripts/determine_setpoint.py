from nn_laser_stabilizer.hardware.connection import create_connection
from nn_laser_stabilizer.connection.pid_connection import ConnectionToPid, LoggingConnectionToPid
from nn_laser_stabilizer.plant import determine_setpoint
from nn_laser_stabilizer.experiment.decorator import experiment
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.logger import AsyncFileLogger


@experiment("setpoint_determination")
def main(context: ExperimentContext) -> None:
    env_args = context.config.env.args
    
    port = env_args.port
    timeout = env_args.timeout
    baudrate = env_args.baudrate
    steps = env_args.setpoint_determination_steps
    max_value = env_args.setpoint_determination_max_value
    factor = env_args.setpoint_determination_factor
    
    connection = create_connection(
        port=port,
        timeout=timeout,
        baudrate=baudrate,
    )
    connection_logger = AsyncFileLogger(
        log_dir='.',
        log_file="connection.log",
    )
    pid_connection = LoggingConnectionToPid(
        connection_to_pid=ConnectionToPid(connection=connection),
        logger=connection_logger,
    )
    
    try:
        pid_connection.open()
        
        context.logger.log(f"Определение setpoint: steps={steps} max_value={max_value} factor={factor}")
        context.logger.log("Выполняется сканирование...")
        
        setpoint, min_pv, max_pv = determine_setpoint(
            pid_connection=pid_connection,
            steps=steps,
            max_value=max_value,
            factor=factor,
        )
        
        context.logger.log(f"Результат: setpoint={setpoint} min_pv={min_pv} max_pv={max_pv}")
        
    finally:
        pid_connection.close()
        connection_logger.close()


if __name__ == "__main__":
    main()
