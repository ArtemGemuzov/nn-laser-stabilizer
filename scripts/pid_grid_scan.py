import itertools
import os
import time
from typing import Iterable
from logging import getLogger

import hydra
from omegaconf import DictConfig

from nn_laser_stabilizer.connection.serial_connection import SerialConnection
from nn_laser_stabilizer.connection.mock_serial_connection import MockSerialConnection
from nn_laser_stabilizer.connection.connection_to_pid import ConnectionToPid
from nn_laser_stabilizer.logging.async_file_logger import AsyncFileLogger
from nn_laser_stabilizer.config.find_configs_dir import find_configs_dir
from nn_laser_stabilizer.config.paths import get_hydra_output_dir
from nn_laser_stabilizer.envs.control_limit_manager import ControlLimitManager, ControlLimitConfig
from nn_laser_stabilizer.envs.constants import (
    KP_MIN,
    DEFAULT_KP,
    KP_MAX,
    KI_MIN,
    DEFAULT_KI,
    KI_MAX,
    KD_MIN,
    DEFAULT_KD,
    KD_MAX,
)


def format_send_log(step: int, time_val: float, kp: float, ki: float, kd: float, u_min: float, u_max: float) -> str:
    return f"step={step} time={time_val:.8f} send kp={kp:.4f} ki={ki:.4f} kd={kd:.4f} u_min={u_min:.4f} u_max={u_max:.4f}"

def format_recv_log(step: int, time_val: float, process_variable: float, control_output: float) -> str:
    return f"step={step} time={time_val:.8f} recv process_variable={process_variable:.4f} control_output={control_output:.4f}"


def linspace_inclusive(start: float, stop: float, num: int) -> Iterable[float]:
    if num <= 1:
        yield stop
        return
    step = (stop - start) / (num - 1)
    for i in range(num):
        yield start + i * step


def valid_combinations(
    n: int,
) -> tuple[Iterable[tuple[float, float, float]], int]:
    kp_values = list(linspace_inclusive(KP_MIN, KP_MAX, n))
    ki_values = list(linspace_inclusive(KI_MIN, KI_MAX, n))
    kd_values = list(linspace_inclusive(KD_MIN, KD_MAX, n))

    kp_inner = [v for v in kp_values if v != KP_MIN and v != KP_MAX]
    ki_inner = [v for v in ki_values if v != KI_MIN and v != KI_MAX]
    kd_inner = [v for v in kd_values if v != KD_MIN and v != KD_MAX]
    total = len(kp_inner) * len(ki_inner) * len(kd_inner)
    return itertools.product(kp_inner, ki_inner, kd_inner), total


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

@hydra.main(config_path=find_configs_dir(), config_name="pid_scan", version_base=None)
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
        steps_per_point = int(config.grid.steps_per_point)

        combos_iter, total_valid = valid_combinations(int(config.grid.num_points))

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
            u_max = float(control_limit_config.default_max)
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

        logger.info("Start scanning: %d combinations", total_valid)
        for idx, (kp, ki, kd) in enumerate(combos_iter, start=1):
            logger.info("combination %d/%d: kp=%.4f ki=%.4f kd=%.4f", idx, total_valid, kp, ki, kd)

            for _ in range(steps_per_point):
                u_min, u_max = control_limit_manager.get_limits_for_step()
                process_variable, control_output = do_exchange(
                    step=step,
                    kp=kp,
                    ki=ki,
                    kd=kd,
                    u_min=u_min,
                    u_max=u_max,
                    pid=pid,
                    interaction_logger=interaction_logger,
                )
                control_limit_manager.apply_rule(control_output)
                step += 1

    except Exception as ex:
        logger.error("Error", ex)

    finally:
        interaction_logger.close()
        connection.close_connection()


if __name__ == "__main__":
    main()


