"""
Скрипт для тестирования соединения с фазовращателем.

Использование:
    python scripts/connection_test.py
    python scripts/connection_test.py --config connection_test -l
    python scripts/connection_test.py --config connection_test --enable-logging
"""

import argparse
import time
import numpy as np

from nn_laser_stabilizer.hardware.connection import create_connection
from nn_laser_stabilizer.connection.phase_shifter_connection import (
    ConnectionToPhaseShifter,
    LoggingConnectionToPhaseShifter,
)
from nn_laser_stabilizer.experiment.decorator import experiment
from nn_laser_stabilizer.experiment.context import ExperimentContext
from nn_laser_stabilizer.logger import SyncFileLogger


@experiment("connection_test")
def main(context: ExperimentContext):
    connection_config = context.config.connection
    port = connection_config.port
    timeout = connection_config.timeout
    baudrate = connection_config.baudrate

    test_config = context.config.test
    control_value = test_config.control_value
    num_iterations = test_config.num_iterations
  
    base_connection = create_connection(
        port=port,
        timeout=timeout,
        baudrate=baudrate,
    )
    
    connection_logger = SyncFileLogger(
        log_dir='.',
        log_file="connection.log",
    )
    phase_shifter = LoggingConnectionToPhaseShifter(
        connection_to_phase_shifter=ConnectionToPhaseShifter(connection=base_connection),
        logger=connection_logger,
    )
       
    context.logger.log(f"Подключение к порту: {port}")
    context.logger.log(f"Отправка постоянного напряжения: {control_value}")
    context.logger.log(f"Количество итераций: {num_iterations}")
    
    exchange_times = np.zeros(num_iterations, dtype=np.float64)
    
    try:
        phase_shifter.open()
        
        for i in range(num_iterations):
            start = time.perf_counter()
            phase_shifter.exchange(control=control_value)
            end = time.perf_counter()
            exchange_times[i] = (end - start) * 1e6  # в микросекундах
    
    except Exception as e:
        context.logger.log(f"Ошибка: {e}")
        raise
    
    finally:
        phase_shifter.close()
        if connection_logger:
            connection_logger.close()
        
        times_file = "exchange_times.txt"
        np.savetxt(times_file, exchange_times, fmt='%.6f', header='Exchange time (microseconds)')
        
        mean_time = np.mean(exchange_times)
        median_time = np.median(exchange_times)
        std_time = np.std(exchange_times)
        min_time = np.min(exchange_times)
        max_time = np.max(exchange_times)
        
        context.logger.log("\n" + "=" * 60)
        context.logger.log("Статистика:")
        context.logger.log(f"  Всего обменов: {num_iterations}")
        context.logger.log(f"  Среднее время: {mean_time:.3f} мкс")
        context.logger.log(f"  Медиана: {median_time:.3f} мкс")
        context.logger.log(f"  Стандартное отклонение: {std_time:.3f} мкс")
        context.logger.log(f"  Минимум: {min_time:.3f} мкс")
        context.logger.log(f"  Максимум: {max_time:.3f} мкс")
        context.logger.log("=" * 60)
        
        context.logger.log("Соединение закрыто")


if __name__ == "__main__":
    main()
