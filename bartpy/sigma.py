

class Sigma:

    def __init__(self, alpha: float, beta: float, scaling_factor):
        self.alpha = alpha
        self.beta = beta
        self._current_value = 1.0
        self.scaling_factor = scaling_factor

    def set_value(self, value: float) -> None:
        self._current_value = value

    def current_value(self) -> float:
        return self._current_value

    def current_unnormalized_value(self) -> float:
        return self.current_value() * self.scaling_factor
