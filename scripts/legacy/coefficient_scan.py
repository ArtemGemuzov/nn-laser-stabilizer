import time
import numpy as np
from enum import Enum
from logging import getLogger

import hydra
from omegaconf import DictConfig

from nn_laser_stabilizer.connection import SerialConnection, MockSerialConnection, ConnectionToPid
from nn_laser_stabilizer.logging import AsyncFileLogger
from nn_laser_stabilizer.config import find_configs_dir, get_hydra_output_dir


class InteractionType(Enum):
    FIRST_WARMUP = "first_warmup"
    FIRST_EXP = "first_exp"
    SECOND_WARMUP = "second_warmup"
    SECOND_EXP = "second_exp"


def format_send_log(step: int, kp: float, ki: float, kd: float, u_min: float, u_max: float, interaction_type: InteractionType) -> str:
    return f"step={step} type={interaction_type.value} send kp={kp:.4f} ki={ki:.4f} kd={kd:.4f} u_min={u_min:.4f} u_max={u_max:.4f}"


def format_recv_log(step: int, process_variable: float, control_output: float, interaction_type: InteractionType) -> str:
    return f"step={step} type={interaction_type.value} recv process_variable={process_variable:.4f} control_output={control_output:.4f}"


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
    interaction_type: InteractionType,
) -> tuple[float, float]:
    """Выполняет обмен данными с PID контроллером."""
    interaction_logger.log(
        format_send_log(step, kp, ki, kd, u_min, u_max, interaction_type)
    )
    process_variable, control_output = pid.exchange(
        kp=kp,
        ki=ki,
        kd=kd,
        control_min=u_min,
        control_max=u_max,
    )
    interaction_logger.log(
        format_recv_log(step, process_variable, control_output, interaction_type)
    )
    return process_variable, control_output


@hydra.main(config_path=find_configs_dir(), config_name="coefficient_scan", version_base=None)
def main(config: DictConfig) -> None:
    logger = getLogger(__name__)
    interaction_logger = AsyncFileLogger(log_dir=get_hydra_output_dir("logs"))
    
    WARMUP_STEPS = 1000
    KP_VALUES = [kp for kp in np.arange(13.5, 1.0, -0.5)]
    KI_VALUE = 11.0
    KD_NUM_POINTS = 50
    KD_MIN = 0.0
    KD_MAX = 0.01
    STEPS_PER_COMBINATION = 200

    DEFAULT_KP = 3.5
    DEFAULT_KI = 11.0
    DEFAULT_KD = 0.02
    
    U_MIN = 0
    U_MAX = 4095
    U_MIN_FORCED = 2000
    U_MAX_FORCED = 2500
    
    try:
        step = 0

        if config.serial.use_mock:
            connection = MockSerialConnection(
                port=config.serial.port,
                timeout=config.serial.timeout,
                baudrate=config.serial.baudrate,
            )
            connection.open()
        else:
            connection = SerialConnection(
                port=config.serial.port,
                timeout=config.serial.timeout,
                baudrate=config.serial.baudrate,
            )
            connection.open()

        pid = ConnectionToPid(connection)

        logger.info("First warmup started")
        for _ in range(WARMUP_STEPS):
            _, _ = do_exchange(
                step=step,
                kp=DEFAULT_KP,
                ki=DEFAULT_KI,
                kd=DEFAULT_KD,
                u_min=U_MIN_FORCED,
                u_max=U_MAX_FORCED,
                pid=pid,
                interaction_logger=interaction_logger,
                interaction_type=InteractionType.FIRST_WARMUP,
            )
            step += 1

        logger.info("First experiment started")
        
        kd_values = np.linspace(KD_MIN, KD_MAX, KD_NUM_POINTS)
        
        total_combinations = len(KP_VALUES) * KD_NUM_POINTS
        combination_count = 0
        
        for kp_idx, kp in enumerate(KP_VALUES, start=1):
            logger.info(f"kp value {kp_idx}/{len(KP_VALUES)}: kp={kp:.4f}")
            
            for _, kd in enumerate(kd_values, start=1):
                combination_count += 1
                logger.info(f"combination {combination_count}/{total_combinations}: kp={kp:.4f} ki={KI_VALUE:.4f} kd={kd:.6f}")

                for _ in range(STEPS_PER_COMBINATION):
                    _, _ = do_exchange(
                        step=step,
                        kp=kp,
                        ki=KI_VALUE,
                        kd=kd,
                        u_min=U_MIN,
                        u_max=U_MAX,
                        pid=pid,
                        interaction_logger=interaction_logger,
                        interaction_type=InteractionType.FIRST_EXP,
                    )
                    step += 1

        logger.info("First experiment completed")

        logger.info("Second warmup started")
        for _ in range(WARMUP_STEPS):
            _, _ = do_exchange(
                step=step,
                kp=DEFAULT_KP,
                ki=DEFAULT_KI,
                kd=DEFAULT_KD,
                u_min=U_MIN_FORCED,
                u_max=U_MAX_FORCED,
                pid=pid,
                interaction_logger=interaction_logger,
                interaction_type=InteractionType.SECOND_WARMUP,
            )
            step += 1

        KP_SECOND_VALUES = np.linspace(1.5, 13.5, 100)
        STEPS_PER_KP_SECOND = 200
        
        logger.info("Second experiment started")
        
        for kp_idx, kp in enumerate(KP_SECOND_VALUES, start=1):
            logger.info(f"kp value {kp_idx}/{len(KP_SECOND_VALUES)}: kp={kp:.4f} ki=0.0 kd=0.0")

            for _ in range(STEPS_PER_KP_SECOND):
                _, _ = do_exchange(
                    step=step,
                    kp=kp,
                    ki=0.0,
                    kd=0.0,
                    u_min=U_MIN,
                    u_max=U_MAX,
                    pid=pid,
                    interaction_logger=interaction_logger,
                    interaction_type=InteractionType.SECOND_EXP,
                )
                step += 1

        logger.info("Second experiment completed")

    except Exception as ex:
        logger.error("Error during scanning", exc_info=True)

    finally:
        interaction_logger.close()
        connection.close()


if __name__ == "__main__":
    main()
