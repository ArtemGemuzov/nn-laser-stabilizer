import signal

from nn_laser_stabilizer.hardware.connection import create_connection
from nn_laser_stabilizer.connection.phase_shifter_connection import (
    ConnectionToPhaseShifter,
    LoggingConnectionToPhaseShifter,
)
from nn_laser_stabilizer.experiment.decorator import experiment
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.logger import SyncFileLogger
from nn_laser_stabilizer.pid import PID


_running = True


def signal_handler(signum, frame):
    global _running
    print("\nОстановка...")
    _running = False


@experiment(
    experiment_name="run-pid-v1",
    config_name="pid_run",
)
def main(context: ExperimentContext):
    global _running
    
    signal.signal(signal.SIGINT, signal_handler)
    
    connection_config = context.config.connection
    port = str(connection_config.port)
    timeout = float(connection_config.timeout)
    baudrate = int(connection_config.baudrate)
    log_connection = bool(connection_config.log_connection)
    
    pid_config = context.config.pid
    kp = float(pid_config.kp)
    ki = float(pid_config.ki)
    kd = float(pid_config.kd)
    dt = float(pid_config.dt)
    min_output = float(pid_config.min_output)
    max_output = float(pid_config.max_output)
    
    setpoint = float(context.config.setpoint)
    
    num_steps = int(context.config.num_steps)
    warmup_steps = int(context.config.warmup_steps)
    warmup_output = int(context.config.warmup_output)
    
    pid = PID(
        kp=kp,
        ki=ki,
        kd=kd,
        dt=dt,
        min_output=min_output,
        max_output=max_output,
    )
    
    base_connection = create_connection(
        port=port,
        timeout=timeout,
        baudrate=baudrate,
    )
    
    phase_shifter = ConnectionToPhaseShifter(connection=base_connection)
    connection_logger = None
    if log_connection:
        connection_logger = SyncFileLogger(
            log_dir=connection_config.log_dir,
            log_file=connection_config.log_file,
        )
        phase_shifter = LoggingConnectionToPhaseShifter(
            connection_to_phase_shifter=phase_shifter,
            logger=connection_logger,
        )
    
    try:
        phase_shifter.open()
        context.logger.log("Соединение открыто")
        
        if warmup_steps > 0:
            pid.min_output = warmup_output
            pid.max_output = warmup_output
        
        context.logger.log("Запуск PID...")

        step = 0
        control_output = warmup_output
        
        while _running:
            process_variable = phase_shifter.exchange(control_output=control_output)
            
            # TODO: коэффициенты подобраны под работу с масштабом /10
            control_output = int(pid(process_variable / 10, setpoint / 10))
            
            step += 1
            
            if step == warmup_steps:
                pid.min_output = min_output
                pid.max_output = max_output
            
            if num_steps > 0 and step >= num_steps:
                break
    
    except Exception as e:
        context.logger.log(f"Ошибка: {e}")
        raise
    
    finally:
        phase_shifter.close()
        
        if connection_logger:
            connection_logger.close()
        
        context.logger.log("Соединение закрыто")


if __name__ == "__main__":
    main()
