from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from nn_laser_stabilizer.oscillator import DuffingOscillator
from nn_laser_stabilizer.pid_controller import PIDController
from nn_laser_stabilizer.numerical_experimental_setup import NumericalExperimentalSetup


def simulate_oscillator(T: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    oscillator = DuffingOscillator()
    controller = PIDController(setpoint=1.0)
    setup = NumericalExperimentalSetup(oscillator, controller, dt=dt)

    np.random.seed(42)
    setup.set_seed(42)

    _, _, setpoint = setup.reset()

    n_steps = int(T / dt)
    times = np.arange(0, T, dt)
    positions = np.zeros(n_steps)
    controls = np.zeros(n_steps)
    setpoints = np.zeros(n_steps)

    kp = 50.0
    ki = 20.0
    kd = 1.0

    for i in range(n_steps):
        position, control, setpoint = setup.step(kp, ki, kd)

        positions[i] = position
        controls[i] = control
        setpoints[i] = setpoint

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(times, positions, label='Положение (x)', color='blue')
    ax1.plot(times, setpoints, label='Установленное значение', color='red', linestyle='-.')
    ax1.set_ylabel('Значение')
    ax1.set_title('Эволюция положения')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(times, controls, label='Управление', color='green', linestyle='--')
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Управление')
    ax2.set_title('Эволюция управления')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return times, positions, controls, setpoints

if __name__ == '__main__':
    simulate_oscillator(dt=0.01, T=100)