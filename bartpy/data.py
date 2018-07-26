class Data:

    def __init__(self, data):
        self.unique_values_cache = {}
        self.data = data

    def variables(self):
        return set(self.data.columns)

    def unique_values(self, variable: str):
        if variable not in self.unique_values_cache:
            self.unique_values_cache[variable] = set(self.data[variable])
        return self.unique_values_cache[variable]
