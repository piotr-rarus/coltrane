from dataclasses import dataclass
from typing import Dict, List

from lazy import lazy
from sklearn.pipeline import Pipeline


@dataclass(init=True)
class SplitStats():
    scores: Dict[str, float]
    pipeline: Pipeline
    dt_fit: float


@dataclass()
class BatchStats():
    splits: List[SplitStats]

    @lazy
    def grouped_scores(self) -> Dict[str, List[float]]:
        """
        Groups evaluated scores from each split by name.

        Returns
        -------
        Dict[str, List[float]]
            [description]
        """

        if not self.splits:
            return {}

        scores = [split.scores for split in self.splits]

        grouped = {}

        for split_scores in scores:
            for metric, value in split_scores.items():

                if metric not in grouped:
                    grouped[metric] = []

                grouped[metric].append(value)

        return grouped
