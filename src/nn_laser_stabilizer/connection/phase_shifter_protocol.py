class PhaseShifterProtocol:
    @staticmethod
    def format_command(control_output: int) -> str:
        return f"{int(control_output)}\n"

    @staticmethod
    def parse_command(command: str) -> int:
        parts = command.strip().split()
        if len(parts) != 1:
            raise ValueError(
                f"Invalid PhaseShifter command format: expected 1 value, got {len(parts)}"
            )
        
        try:
            control = int(parts[0])
        except ValueError as e:
            raise ValueError(f"Error parsing PhaseShifter command values: {e}")
        return control

    @staticmethod
    def format_response(process_variable: int) -> str:
        return f"{process_variable}\n"

    @staticmethod
    def parse_response(response: str) -> int:
        parts = response.strip().split()
        if len(parts) != 1:
            raise ValueError(
                f"Invalid PhaseShifter response format: expected 1 value, got {len(parts)}"
            )
        
        try:
            pv = int(parts[0])
        except ValueError as e:
            raise ValueError(f"Error parsing PhaseShifter response values: {e}")
        return pv

