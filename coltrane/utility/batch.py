from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import numpy as np
from lazy_property import LazyProperty

from . import split


@dataclass()
class Stats():
    splits: List[split.Stats]

    def __init__(self):
        self.splits = []

    def get_aggregated_performance(self):

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

        aggregated[dt_fit.__name__] = self.__aggregate_stat(dt_fit)
        aggregated[dt_predict.__name__] = self.__aggregate_stat(dt_predict)

        aggregated[dt_predict_record.__name__] = self.__aggregate_stat(
            dt_predict_record
        )

        return aggregated

    def get_aggregated_metrics(self):
        aggregated = OrderedDict()

        for metric, values in self.grouped_metrics.items():
            stats = OrderedDict()

            stats['mean'] = np.mean(values)
            stats['min'] = np.min(values)
            stats['max'] = np.max(values)
            stats['std'] = np.std(values)

            aggregated[metric] = stats

        return aggregated

    @LazyProperty
    def grouped_metrics(self, metrics):
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
