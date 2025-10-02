import time
import random
from typing import Iterable
from logging import getLogger

import hydra
from omegaconf import DictConfig

from nn_laser_stabilizer.connection.serial_connection import SerialConnection
from nn_laser_stabilizer.connection.mock_serial_connection import MockSerialConnection
from nn_laser_stabilizer.connection.connection_to_pid import ConnectionToPid
from nn_laser_stabilizer.logging.async_file_logger import AsyncFileLogger
from nn_laser_stabilizer.config.find_configs_dir import find_configs_dir
from nn_laser_stabilizer.config.hydra_paths import get_hydra_output_dir
from nn_laser_stabilizer.envs.control_limit_manager import ControlLimitManager, ControlLimitConfig
from nn_laser_stabilizer.envs.constants import (
    KP_MIN,
    DEFAULT_KP,
    KP_MAX,
    DEFAULT_KI,
    DEFAULT_KD,
)


def format_send_log(step: int, time_val: float, kp: float, ki: float, kd: float, u_min: float, u_max: float) -> str:
    return f"step={step} time={time_val:.8f} send kp={kp:.4f} ki={ki:.4f} kd={kd:.4f} u_min={u_min:.4f} u_max={u_max:.4f}"

def format_recv_log(step: int, time_val: float, process_variable: float, control_output: float) -> str:
    return f"step={step} time={time_val:.8f} recv process_variable={process_variable:.4f} control_output={control_output:.4f}"


def linspace_inclusive(start: float, stop: float, num: int) -> Iterable[float]:
    """Генерирует num равномерно распределенных значений от start до stop включительно."""
    if num <= 1:
        yield stop
        return
    step = (stop - start) / (num - 1)
    for i in range(num):
        yield start + i * step


def kp_values(n: int) -> Iterable[float]:
    """Генерирует n значений kp между KP_MIN и KP_MAX, исключая границы."""
    # Генерируем n+2 точек включительно, затем исключаем границы
    all_values = list(linspace_inclusive(KP_MIN, KP_MAX, n + 2))
    # Исключаем первую (KP_MIN) и последнюю (KP_MAX) точки
    return all_values[1:-1]


def do_exchange(
    *,
    step: int,
    kp: float,
    ki: float,
    kd: float,
    u_min: float,
    u_max: float,
    pid: ConnectionToPid,
    interaction_logger: AsyncFileLogger,
) -> tuple[float, float]:
    """Выполняет обмен данными с PID контроллером."""
    interaction_logger.log(
        format_send_log(step, time.time(), kp, ki, kd, u_min, u_max)
    )
    process_variable, control_output = pid.send_and_read(
        kp=kp,
        ki=ki,
        kd=kd,
        control_min=u_min,
        control_max=u_max,
    )
    interaction_logger.log(
        format_recv_log(step, time.time(), process_variable, control_output)
    )
    return process_variable, control_output


@hydra.main(config_path=find_configs_dir(), config_name="kp_scan", version_base=None)
def main(config: DictConfig) -> None:
    logger = getLogger(__name__)
    interaction_logger = AsyncFileLogger(log_dir=get_hydra_output_dir("logs"))
    
    try:
        control_limit_config = ControlLimitConfig(
            default_min=float(config.control_output_limits.default_min),
            default_max=float(config.control_output_limits.default_max),
            force_min_value=float(config.control_output_limits.force_min_value),
            force_max_value=float(config.control_output_limits.force_max_value),
            lower_force_condition_threshold=float(config.control_output_limits.lower_force_condition_threshold),
            upper_force_condition_threshold=float(config.control_output_limits.upper_force_condition_threshold),
            enforcement_steps=int(config.control_output_limits.enforcement_steps),
        )
        enforcement_steps = control_limit_config.enforcement_steps
        control_limit_manager = ControlLimitManager(control_limit_config)

        step = 0
        steps_per_kp_value = int(config.kp_scan.steps_per_kp_value)
        num_kp_points = int(config.kp_scan.num_points)

        kp_values_list = list(kp_values(num_kp_points))

        if config.serial.use_mock:
            connection = MockSerialConnection(
                port=config.serial.port,
                timeout=config.serial.timeout,
                baudrate=config.serial.baudrate,
            )
            connection.open_connection()
        else:
            connection = SerialConnection(
                port=config.serial.port,
                timeout=config.serial.timeout,
                baudrate=config.serial.baudrate,
            )
            connection.open_connection()

        pid = ConnectionToPid(connection)

        logger.info(f"Warmup: sending {enforcement_steps} default coefficients with elevated control limits")
        for _ in range(enforcement_steps):
            u_min = float(control_limit_config.force_min_value)
            u_max = float(control_limit_config.force_max_value)
            process_variable, control_output = do_exchange(
                step=step,
                kp=DEFAULT_KP,
                ki=DEFAULT_KI,
                kd=DEFAULT_KD,
                u_min=u_min,
                u_max=u_max,
                pid=pid,
                interaction_logger=interaction_logger,
            )
            control_limit_manager.apply_rule(control_output)
            step += 1

        logger.info("Starting kp scan (ascending): %d values", len(kp_values_list))
        for idx, kp in enumerate(kp_values_list, start=1):
            logger.info("kp value %d/%d: kp=%.4f (ki=%.4f, kd=%.4f fixed) - ascending", 
                       idx, len(kp_values_list), kp, DEFAULT_KI, DEFAULT_KD)

            steps_completed = 0
            while steps_completed < steps_per_kp_value:
                u_min, u_max = control_limit_manager.get_limits_for_step()
                process_variable, control_output = do_exchange(
                    step=step,
                    kp=kp,
                    ki=DEFAULT_KI,  
                    kd=DEFAULT_KD,  
                    u_min=u_min,
                    u_max=u_max,
                    pid=pid,
                    interaction_logger=interaction_logger,
                )
                control_limit_manager.apply_rule(control_output)
                step += 1
                
                if control_limit_manager._force_steps_left == 0:
                    steps_completed += 1
                else:
                    logger.info("Force limits active, step not counted (force_steps_left=%d)", 
                               control_limit_manager._force_steps_left)

        logger.info("Ascending scan completed. Starting warmup before descending scan...")
        
        for _ in range(enforcement_steps):
            u_min = float(control_limit_config.force_min_value)
            u_max = float(control_limit_config.force_max_value)
            process_variable, control_output = do_exchange(
                step=step,
                kp=DEFAULT_KP,
                ki=DEFAULT_KI,
                kd=DEFAULT_KD,
                u_min=u_min,
                u_max=u_max,
                pid=pid,
                interaction_logger=interaction_logger,
            )
            control_limit_manager.apply_rule(control_output)
            step += 1

        logger.info("Starting kp scan (descending): %d values", len(kp_values_list))
        kp_values_descending = list(reversed(kp_values_list))
        for idx, kp in enumerate(kp_values_descending, start=1):
            logger.info("kp value %d/%d: kp=%.4f (ki=%.4f, kd=%.4f fixed) - descending", 
                       idx, len(kp_values_descending), kp, DEFAULT_KI, DEFAULT_KD)

            steps_completed = 0
            while steps_completed < steps_per_kp_value:
                u_min, u_max = control_limit_manager.get_limits_for_step()
                process_variable, control_output = do_exchange(
                    step=step,
                    kp=kp,
                    ki=DEFAULT_KI,  
                    kd=DEFAULT_KD, 
                    u_min=u_min,
                    u_max=u_max,
                    pid=pid,
                    interaction_logger=interaction_logger,
                )
                control_limit_manager.apply_rule(control_output)
                step += 1
                
                if control_limit_manager._force_steps_left == 0:
                    steps_completed += 1
                else:
                    logger.info("Force limits active, step not counted (force_steps_left=%d)", 
                               control_limit_manager._force_steps_left)

        logger.info("kp scan completed (both ascending and descending)")
        
        logger.info("Starting additional warmup with default coefficients...")
        for _ in range(enforcement_steps):
            u_min = float(control_limit_config.force_min_value)
            u_max = float(control_limit_config.force_max_value)
            process_variable, control_output = do_exchange(
                step=step,
                kp=DEFAULT_KP,
                ki=DEFAULT_KI,
                kd=DEFAULT_KD,
                u_min=u_min,
                u_max=u_max,
                pid=pid,
                interaction_logger=interaction_logger,
            )
            control_limit_manager.apply_rule(control_output)
            step += 1
        
        logger.info("Starting Wiener process coefficient variation...")
        
        wiener_steps = 10_000
        wiener_noise_scale = KP_MAX / 100
        
        current_kp = DEFAULT_KP
        current_ki = DEFAULT_KI
        current_kd = DEFAULT_KD
        
        kp_wiener_min = KP_MIN
        kp_wiener_max = KP_MAX
        
        for wiener_step in range(wiener_steps):
            kp_noise = random.gauss(0, wiener_noise_scale)
            
            current_kp += kp_noise
            
            current_kp = max(kp_wiener_min, min(kp_wiener_max, current_kp))
            
            if wiener_step % 1000 == 0:
                logger.info("Wiener step %d/%d: kp=%.4f ki=%.4f kd=%.4f", 
                        wiener_step + 1, wiener_steps, current_kp, current_ki, current_kd)
            
            u_min, u_max = control_limit_manager.get_limits_for_step()
            process_variable, control_output = do_exchange(
                step=step,
                kp=current_kp,
                ki=current_ki,
                kd=current_kd,
                u_min=u_min,
                u_max=u_max,
                pid=pid,
                interaction_logger=interaction_logger,
            )
            control_limit_manager.apply_rule(control_output)
            step += 1
        
        logger.info("Wiener process coefficient variation completed")

    except Exception as ex:
        logger.error("Error during scanning", exc_info=True)

    finally:
        interaction_logger.close()
        connection.close_connection()


if __name__ == "__main__":
    main()
