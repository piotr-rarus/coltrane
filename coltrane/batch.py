import json
from dataclasses import dataclass
from hashlib import blake2b
from typing import Dict

from lazy import lazy
from sklearn.base import TransformerMixin
from sklearn.metrics._scorer import _BaseScorer
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from coltrane.file.io.base import Data


@dataclass(frozen=True)
class Batch:
    """
    Define your research with this class.
    """

    data: Data
    pipeline: Pipeline
    selection: BaseCrossValidator
    scorers: Dict[str, _BaseScorer]
    encoder: TransformerMixin = None
    multiprocessing: bool = False

    @lazy
    def as_nice_hash(self) -> str:
        """
        Short identifier of the batch. Logs will appear under it.

        Returns
        -------
        str
        """

        encoded = str(id(self)).encode("utf-8")
        return blake2b(encoded, digest_size=4).hexdigest()

    @lazy
    def as_dict(self):
        """
        Batch config summary.

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

        for name, metric in self.scorers.items():
            metrics[name] = metric._kwargs

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
