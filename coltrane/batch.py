import json
from typing import Generator, List, Tuple

from sklearn.pipeline import Pipeline
from tqdm import tqdm

from .file.io.base import DataSet


class Batch():

    def __init__(
        self,
        *,
        data_set: DataSet,
        pipelines: Generator[Pipeline, None, None],
        selection,
        metrics: List[Tuple]
    ):

        self.data_set = data_set
        self.selection = selection
        self.pipelines = pipelines
        self.metrics = metrics

    def as_dict(self):
        """
        Pipeline's config summary.

        Returns
        -------
        Dictionary
            Outputs nicely written pipeline configuration.
        """

        batch = {}
        batch['data_set'] = self.data_set.as_dict()

        batch['selection'] = {
            'cv': self.selection.__class__.__name__,
            'cvargs': self.selection.cvargs,
            'n_repeats': self.selection.n_repeats,
            'random_state': self.selection.random_state
        }

        metrics = {}

        for metric in self.metrics:
            if type(metric) is tuple:
                op, kwargs = metric
                metrics[op.__name__] = kwargs
            else:
                metrics[metric.__name__] = {}

        batch['metrics'] = metrics

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
        tqdm.write(json.dumps(self.as_dict(), indent=4))
        tqdm.write('\n' * 3)
