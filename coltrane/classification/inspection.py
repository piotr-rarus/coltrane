from typing import Dict

import numpy as np
from austen import Logger

from coltrane.file.io.base import Data
from coltrane.inspection import Inspector as Base


class Inspector(Base):

    def __init__(self):
        super(Inspector, self).__init__()

    def __post_inspect(self, data: Data, logger: Logger):

        summary = {}
        balance = self.get_class_balance(data)
        summary['balance'] = balance

        return summary

    def get_class_balance(self, data: Data) -> Dict[str, int]:
        """
        Computes class balance in data set.

        Parameters
        ----------
        data : Data
            Single label data set.

        Returns
        -------
        Dict[str, int]
            Record count for each label.
        """

        labels = data.y

        class_balance = {}
        aggregate = np.unique(labels, return_counts=True)

        for label, count in np.transpose(aggregate):
            class_balance[str(label)] = count

        return class_balance

    def plot_class_balance(self, balance: Dict[str, int]):
        """
        Plots class balance into the notebook.

        Parameters
        ----------
        balance : Dict[str, int]
            Record count for each label.
            Use `get_class_balance` to obtain this dictionary.
        """

        self.plot.class_balance(balance)

    def plot_features_distribution(self, data: Data):
        """
        Plots features distribution using PCA to reduce dimensionality.

        Parameters
        ----------
        data : Data
            Single label data set.
        """

        self.plot.features_distribution(data.x, data.y)
