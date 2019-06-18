from collections import OrderedDict
from lazy_property import LazyProperty

# from .extension import tuple_ext
from .file.io.base import DataSet

from sklearn.pipeline import Pipeline


class Batch():
    # ! document this shit

    def __init__(
        self,
        *,
        data: DataSet,
        pipeline: Pipeline,
        selection,
        validation
    ):

        self.data = data
        self.selection = selection
        self.pipeline = pipeline
        self.validation = validation

    @LazyProperty
    def pprint(self):
        """
        Pipeline's config summary.

        Returns
        -------
        Dictionary
            Outputs nicely written pipeline configuration.
        """

        config = OrderedDict()

        config['data'] = self.data.pprint

        config['selection'] = {
            'cv': self.selection.__class__.__name__,
            'cvargs': self.selection.cvargs,
            'n_repeats': self.selection.n_repeats,
            'random_state': self.selection.random_state
        }

        config['pipeline'] = {}

        for step in self.pipeline:
            config['pipeline'][step.__class__.__name__] = 'foo'

        return config
