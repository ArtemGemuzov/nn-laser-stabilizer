import json
import signal
import time

import numpy as np

from nn_laser_stabilizer.hardware.connection import create_connection
from nn_laser_stabilizer.connection.phase_shifter_connection import ConnectionToPhaseShifter
from nn_laser_stabilizer.experiment.decorator import experiment
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.logger import SyncFileLogger
from nn_laser_stabilizer.pid import PIDDelta


_running = True


def signal_handler(signum, frame):
    global _running
    _running = False


@experiment(
    experiment_name="run-pid-v3",
    config_name="pid_run",
)
def main(context: ExperimentContext):
    global _running
    
    signal.signal(signal.SIGINT, signal_handler)
    
    connection_config = context.config.connection
    port = str(connection_config.port)
    timeout = float(connection_config.timeout)
    baudrate = int(connection_config.baudrate)
    
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
    
    noise_config = context.config.noise
    noise_std = float(noise_config.std)
    noise_clip = float(noise_config.clip)
    
    pid_delta = PIDDelta(kp=kp, ki=ki, kd=kd, dt=dt)
    
    base_connection = create_connection(
        port=port,
        timeout=timeout,
        baudrate=baudrate,
    )
    
    phase_shifter = ConnectionToPhaseShifter(connection=base_connection)
    
    logger = SyncFileLogger(
        log_dir=".",
        log_file="pid_data.jsonl",
    )
    
    try:
        phase_shifter.open()
        context.logger.log("Соединение открыто")
        
        context.logger.log("Начало работы ПИД...")
        
        warmup_step = 0
        is_warming_up = warmup_steps > 0 
        control_output = warmup_output
        step = 0

        prev_time = time.perf_counter()
        
        while _running:
            now = time.perf_counter()
            delta_time = now - prev_time
            prev_time = now

            process_variable = phase_shifter.exchange(control_output=control_output)
            
            # TODO: коэффициенты подобраны под работу с масштабом /10
            delta = pid_delta(process_variable / 10, setpoint / 10)
            
            noise = float(np.clip(
                np.random.normal(0, noise_std),
                -noise_clip,
                noise_clip,
            ))
            
            if is_warming_up:
                control_output = warmup_output
                clean_control_output = warmup_output
                warmup_step += 1
                
                if warmup_step >= warmup_steps:
                    is_warming_up = False
            else:
                clean_control_output = int(np.clip(
                    control_output + delta,
                    min_output,
                    max_output,
                ))
                control_output = int(np.clip(
                    clean_control_output + noise,
                    min_output,
                    max_output,
                ))
                
                if control_output >= max_output or control_output <= min_output:
                    context.logger.log(
                        f"Сигнал вышел за пределы ({control_output})"
                    )
                    is_warming_up = True
                    warmup_step = 0
                    control_output = warmup_output
                    clean_control_output = warmup_output
            
            logger.log(json.dumps({
                "step": step,
                "delta_time": delta_time,
                "process_variable": int(process_variable),
                "control_output": int(control_output),
                "clean_control_output": int(clean_control_output),
                "delta": float(delta),
                "noise": noise,
                "is_warming_up": is_warming_up,
            }))
            
            step += 1
            
            if num_steps > 0 and step >= num_steps:
                break
    
    except Exception as e:
        context.logger.log(f"Ошибка: {e}")
        raise
    
    finally:
        phase_shifter.close()
        logger.close()
        
        context.logger.log("Соединение закрыто")


if __name__ == "__main__":
    main()
