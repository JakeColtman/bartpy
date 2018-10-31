from typing import List, Mapping, Union, Any

import numpy as np
from tqdm import tqdm

from bartpy.model import Model, deep_copy_model
from bartpy.samplers.sampler import Sampler
from bartpy.samplers.schedule import SampleSchedule


Chain = Mapping[str, Union[List[Any], np.ndarray]]


class ModelSampler(Sampler):

    def __init__(self, schedule: SampleSchedule):
        self.schedule = schedule

    def step(self, model: Model):
        step_result = {}
        for step in self.schedule.steps(model):
            result = step()
            if result is not None:
                if result[0] not in step_result:
                    step_result[result[0]] = []
                step_result[result[0]].append(result[1])
        return {x: np.mean([1 if y else 0 for y in step_result[x]]) for x in step_result}

    def samples(self, model: Model,
                n_samples: int,
                n_burn: int,
                thin: float=0.1,
                store_in_sample_predictions: bool=True,
                store_acceptance: bool=True) -> Chain:
        print("Starting burn")
        for _ in tqdm(range(n_burn)):
            self.step(model)
        trace = []
        model_trace = []
        acceptance_trace = []
        print("Starting sampling")

        thin_inverse = 1. / thin

        for ss in tqdm(range(n_samples)):
            acceptance_dict = self.step(model)
            if ss % thin_inverse == 0:
                if store_in_sample_predictions:
                    trace.append(model.predict())
                if store_acceptance:
                    acceptance_trace.append(acceptance_dict)
                model_trace.append(deep_copy_model(model))
        return {
            "model": model_trace,
            "acceptance": acceptance_trace,
            "in_sample_predictions": trace
        }
