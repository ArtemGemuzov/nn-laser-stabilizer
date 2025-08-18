from simulate_oscillator import simulate_oscillator

if __name__ == "__main__":
    times, positions, controls, setpoints = simulate_oscillator(T=100, dt=2E-3)
