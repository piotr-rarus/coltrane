from dataclasses import dataclass
from typing import List

from . import aggregate, split


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

        aggregated = {}

        aggregated['dt_fit'] = aggregate.Stats(dt_fit).as_dict()
        aggregated['dt_predict'] = aggregate.Stats(dt_predict).as_dict()
        aggregated['dt_predict_record'] = aggregate.Stats(
            dt_predict_record
        ).as_dict()

        return aggregated

    @property
    def aggregated_metrics(self):

        if not self.splits:
            return {}

        aggregated = {}

        for metric, values in self.grouped_metrics.items():
            aggregated[metric] = aggregate.Stats(values).as_dict()

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
