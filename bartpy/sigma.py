

class Sigma:

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self._current_value = 1.0

    def set_value(self, value: float) -> None:
        self._current_value = value

    def current_value(self) -> float:
        return self._current_value
