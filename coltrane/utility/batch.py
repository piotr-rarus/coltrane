from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import numpy as np

from . import split


@dataclass()
class Stats():
    splits: List[split.Stats]

    def __init__(self):
        self.splits = []

    @property
    def aggregated_performance(self):

        if not self.splits:
            return {}

        performances = [split.performance for split in self.splits]

        dt_fit = [
            performance.dt_fit
            for performance
            in performances
        ]

        dt_predict = [
            performance.dt_predict
            for performance
            in performances
        ]

        dt_predict_record = [
            performance.dt_predict_record
            for performance
            in performances
        ]

        aggregated = OrderedDict()

        aggregated['dt_fit'] = self.__aggregate_stat(dt_fit)
        aggregated['dt_predict'] = self.__aggregate_stat(dt_predict)

        aggregated['dt_predict_record'] = self.__aggregate_stat(
            dt_predict_record
        )

        return aggregated

    @property
    def aggregated_metrics(self):

        if not self.splits:
            return {}

        aggregated = OrderedDict()

        for metric, values in self.grouped_metrics.items():
            stats = OrderedDict()

            stats['mean'] = np.mean(values)
            stats['min'] = np.min(values)
            stats['max'] = np.max(values)
            stats['std'] = np.std(values)

            aggregated[metric] = stats

        return aggregated

    @property
    def grouped_metrics(self):
        """
        Groups evaluation metric values from each split, by metrics.

        Parameters
        ----------
        stats : list[dict]
            List of evaluation stats from each `train/test` split.
            Please follow the structure denoted
            by sklearn.metrics.classification_report.

        Returns
        -------
        dict
            Grouped evaluation measures.
        """

        if not self.splits:
            return {}

        metrics = [split.metrics for split in self.splits]

        grouped = {}

        for stat in metrics:
            for metric, value in stat.items():

                if metric not in grouped:
                    grouped[metric] = []

                grouped[metric].append(value)

        return grouped

    def __aggregate_stat(self, values: List[float]):
        return {
            'mean': np.mean(values),
            'min': np.min(values),
            'max': np.max(values),
            'std': np.std(values)
        }
