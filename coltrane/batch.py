import json
from typing import List, Tuple

from lazy_property import LazyProperty
from sklearn.pipeline import Pipeline
from tqdm import tqdm

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
    def as_dict(self):
        """
        Pipeline's config summary.

        Returns
        -------
        Dictionary
            Outputs nicely written pipeline configuration.
        """

        batch = {}
        batch['data'] = self.data.as_dict

        batch['pipeline'] = {}

        for step in self.pipeline:
            batch['pipeline'][step.__class__.__name__] = vars(step)

        batch['selection'] = {
            'cv': self.selection.__class__.__name__,
            'cvargs': self.selection.cvargs,
            'n_repeats': self.selection.n_repeats,
            'random_state': self.selection.random_state
        }

        return batch

    def pprint(self):
        """
        Pretty prints your data set and pipeline onto console using tqdm.

        Parameters
        ----------
        data_set : DataSet
            Instantiated data set.
        pipeline : Pipeline
            Configured pipeline template.

        """

        tqdm.write('\n' * 3)
        tqdm.write('=' * 100)
        tqdm.write('\n' * 3)

        tqdm.write('Data set:')
        tqdm.write(json.dumps(self.as_dict, indent=4))
        tqdm.write('\n' * 3)
