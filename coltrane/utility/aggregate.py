from dataclasses import dataclass
from typing import List

import numpy as np


"""
Utility functions for model evaluation stats/performance aggregation.

"""


@dataclass
class Stats():
    mean: float
    min: float
    max: float
    std: float

    def __init__(self, values: List[float]):
        self.mean = np.mean(values)
        self.min = np.min(values)
        self.max = np.max(values)
        self.std = np.std(values)

    def as_dict(self):
        return self.__dict__


def balance(labels):
    """
    Calculates class balance stats.

    Parameters
    ----------
    labels : string
        Ground truth prediction

    Returns
    -------
    dict
        Stats.
    """

    summary = {}
    aggregate = np.unique(labels, return_counts=True)

    for label, count in np.transpose(aggregate):
        summary[str(label)] = count

    counts = aggregate[1]

    summary['stats'] = Stats(counts).as_dict()

    return summary


def confusion_matrix(confusion_matrices):
    """
    Computes average values among confusion matrices.

    Parameters
    ----------
    confusion_matrices : list[nd.array]
        List of confusion matrices from each selection split.

    """

    return np.average(confusion_matrices, axis=0)
