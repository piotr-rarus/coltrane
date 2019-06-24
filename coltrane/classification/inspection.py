from collections import OrderedDict

from austen import Logger

from ..file.io.base import DataSet
from ..inspection import Inspector as Base
from ..utility import aggregate, plot


class Inspector(Base):

    def __init__(self):
        return super().__init__()

    def __post_inspect(self, data_set: DataSet, logger: Logger):

        records = data_set.X
        labels = data_set.y

        summary = OrderedDict()
        summary['balance'] = aggregate.balance(labels)
        plot.labels_distribution(labels, logger, 'balance')

        plot.features_distribution(
            records,
            labels,
            logger,
            'distribution'
        )
