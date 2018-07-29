

class Sigma:

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self._current_value = 1.0

    def set_value(self, value: float) -> None:
        if not isinstance(value, float):
            raise TypeError("Sigma can only have float values, found {}".format(type(value)))
        self._current_value = value

    def current_value(self) -> float:
        return self._current_value
