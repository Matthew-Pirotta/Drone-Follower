import time

class PDController:
    def __init__(self, Kp, Kd, setpoint=0.0, min_output=5, max_output=20):
        self.Kp = Kp  # Proportional gain
        self.Kd = Kd  # Derivative gain
        self.setpoint = setpoint  # Desired value
        self.prev_error = 0
        self.last_time = time.time()
        self.min_output = min_output  # Minimum movement threshold
        self.max_output = max_output  # Prevent excessive corrections

    def compute(self, current_value):
        """
        Compute the PD output based on the current error.
        """
        current_time = time.time()
        dt = max(current_time - self.last_time, 0.01)  # Prevent division by zero
        error = self.setpoint - current_value
        derivative = (error - self.prev_error) / dt  # Rate of change of error

        output = (self.Kp * error) + (self.Kd * derivative)

        # Apply minimum movement threshold
        if abs(output) < self.min_output:
            output = 0

        # Clamp output to avoid extreme values
        output = max(min(output, self.max_output), -self.max_output)

        # Update previous values
        self.prev_error = error
        self.last_time = current_time

        return int(output)  # Convert to integer for drone commands
