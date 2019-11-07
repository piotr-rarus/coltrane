from dataclasses import dataclass
from typing import List

import numpy as np
from lazy import lazy


@dataclass(frozen=True)
class Stats:
    values: List[float]

    @lazy
    def percentiles(self):

        percentiles = []

        for p in range(0, 101, 10):
            value_at_p = np.percentile(self.values, p)
            percentiles.append(value_at_p)

        return percentiles

    @lazy
    def mean(self):
        return np.mean(self.values)

    @lazy
    def std(self):
        return np.std(self.values)

    @lazy
    def as_dict(self):
        stats = {}
        stats['mean'] = self.mean
        stats['std'] = self.std
        stats['percentiles'] = self.percentiles

        return stats
