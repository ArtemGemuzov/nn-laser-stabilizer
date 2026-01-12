class PidProtocol:
    KP_DECIMAL_PLACES = 4
    KI_DECIMAL_PLACES = 4
    KD_DECIMAL_PLACES = 6
    
    @staticmethod
    def format_command(
        kp: float,
        ki: float,
        kd: float,
        control_min: int,
        control_max: int,
    ) -> str:
        return (
            f"{kp:.{PidProtocol.KP_DECIMAL_PLACES}f} "
            f"{ki:.{PidProtocol.KI_DECIMAL_PLACES}f} "
            f"{kd:.{PidProtocol.KD_DECIMAL_PLACES}f} "
            f"{control_min} {control_max}\n"
        )

    @staticmethod
    def parse_command(command: str) -> tuple[float, float, float, float, float]:
        parts = command.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid command format: expected 5 values, got {len(parts)}")
        
        try:
            return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError as e:
            raise ValueError(f"Error parsing command values: {e}")

    @staticmethod
    def format_response(process_variable: int, control_output: int) -> str:
        return f"{process_variable} {control_output}\n"

    @staticmethod
    def parse_response(response: str) -> tuple[float, float]:
        parts = response.strip().split()
        if len(parts) != 2:
            raise ValueError(f"Invalid PID response format: {repr(response)}")
        return float(parts[0]), float(parts[1])

