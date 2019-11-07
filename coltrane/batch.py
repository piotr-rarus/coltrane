import json
from dataclasses import dataclass
from hashlib import blake2b
from typing import Dict, List, Tuple

from lazy import lazy
from sklearn.base import TransformerMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from coltrane.file.io.base import Data


@dataclass(frozen=True)
class Batch:

    data: Data
    pipeline: Pipeline
    selection: BaseCrossValidator
    metrics: List[Tuple]
    encoder: TransformerMixin = None
    multiprocessing: bool = False

    @lazy
    def as_nice_hash(self) -> str:
        encoded = str(id(self)).encode("utf-8")
        return blake2b(encoded, digest_size=4).hexdigest()

    @lazy
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
        batch['pipeline'] = self.pipeline_as_dict

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

    @lazy
    def pipeline_as_dict(self) -> Dict:
        as_dict = {}

        for name, step in self.pipeline.steps:
            as_dict[name] = vars(step)

        return as_dict

    def pprint(self):
        """
        Pretty prints your data set and pipeline onto console using tqdm.
        """

        tqdm.write('\n' * 3)
        tqdm.write('=' * 100)
        tqdm.write('\n' * 3)

        tqdm.write('Data set:')
        tqdm.write(json.dumps(self.as_dict, indent=4))
        tqdm.write('\n' * 3)
