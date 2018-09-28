from copy import deepcopy
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from bartpy.model import Model, deep_copy_model
from bartpy.samplers.schedule import SampleSchedule
from bartpy.samplers.sampler import Sampler


class ModelSampler(Sampler):

    def __init__(self, schedule: SampleSchedule):
        self.schedule = schedule

    def step(self, model: Model):
        for step in self.schedule.steps(model):
            step()

    def samples(self, model: Model, n_samples: int, n_burn: int, thin: float=0.1, store_in_sample_predictions: bool=True) -> Tuple[List[Model], np.ndarray]:
        print("Starting burn")
        for _ in tqdm(range(n_burn)):
            self.step(model)
        trace = []
        model_trace = []
        print("Starting sampling")

        thin_inverse = 1. / thin

        for ss in tqdm(range(n_samples)):
            self.step(model)
            if ss % thin_inverse == 0:
                if store_in_sample_predictions:
                    trace.append(model.predict())
                model_trace.append(deep_copy_model(model))
        return model_trace, np.array(trace)