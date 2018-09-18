from copy import deepcopy
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from bartpy.model import Model
from bartpy.samplers.schedule import SampleSchedule


class Sampler:

    def __init__(self, model: Model, schedule: SampleSchedule):
        self.schedule = schedule
        self.model = model

    def step(self):
        for ss in self.schedule.steps():
            ss.step()

    def samples(self, n_samples: int, n_burn: int) -> Tuple[List[Model], np.ndarray]:
        print("Starting burn")
        for _ in tqdm(range(n_burn)):
            self.step()
        trace = []
        model_trace = []
        print("Starting sampling")
        for ss in tqdm(range(n_samples)):
            self.step()
            trace.append(self.model.predict())
            model_trace.append(deepcopy(self.model))
        return model_trace, np.array(trace)