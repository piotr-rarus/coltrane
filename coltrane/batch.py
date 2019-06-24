from collections import OrderedDict
from typing import List, Tuple

from lazy_property import LazyProperty
from sklearn.pipeline import Pipeline

from .file.io.base import DataSet


class Batch():

    def __init__(
        self,
        *,
        data: DataSet,
        pipeline: Pipeline,
        selection,
        metrics: List[Tuple]
    ):

        self.data = data
        self.selection = selection
        self.pipeline = pipeline
        self.metrics = metrics

    @LazyProperty
    def pprint(self):
        """
        Pipeline's config summary.

        Returns
        -------
        Dictionary
            Outputs nicely written pipeline configuration.
        """

        config = {}
        config['data'] = self.data.pprint

        config['selection'] = {
            'cv': self.selection.__class__.__name__,
            'cvargs': self.selection.cvargs,
            'n_repeats': self.selection.n_repeats,
            'random_state': self.selection.random_state
        }

        config['pipeline'] = {}

        for step in self.pipeline:
            config['pipeline'][step.__class__.__name__] = vars(step)

        return config
