from copy import deepcopy

import numpy as np

from bartpy.model import Model
from bartpy.samplers.schedule import SampleSchedule


class Sampler:

    def __init__(self, model: Model, schedule: SampleSchedule):
        self.schedule = schedule
        self.model = model

    def step(self):
        for ss in self.schedule.steps():
            ss.step()

    def samples(self, n_samples: int, n_burn: int) -> np.ndarray:
        for bb in range(n_burn):
            print("Burn - ", bb)
            self.step()
        trace = []
        model_trace = []
        for ss in range(n_samples):
            print("Sample - ", ss)
            self.step()
            trace.append(self.model.predict())
            model_trace.append(deepcopy(self.model))
        return model_trace, np.array(trace)