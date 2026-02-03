from nn_laser_stabilizer.hardware.connection import create_connection
from nn_laser_stabilizer.connection.pid_connection import ConnectionToPid, LoggingConnectionToPid
from nn_laser_stabilizer.envs.setpoint import determine_setpoint
from nn_laser_stabilizer.experiment.decorator import experiment
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.logger import AsyncFileLogger


@experiment(
    experiment_name="determine_setpoint", 
    config_name="setpoint_determination"
)
def main(context: ExperimentContext) -> None:
    env_args = context._config.env.args
    
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
        
        def send_control_and_get_pv(control_value: int) -> int:
            pv, _ = pid_connection.exchange(
                kp=0.0,
                ki=0.0,
                kd=0.0,
                control_min=control_value,
                control_max=control_value,
                setpoint=0,
            )
            return int(pv)

        setpoint, min_pv, max_pv = determine_setpoint(
            send_control_and_get_pv=send_control_and_get_pv,
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
