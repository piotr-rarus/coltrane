from austen import Logger

from coltrane.file.io.base import Data
from coltrane.inspection import Inspector as Base
from coltrane.util import aggregate, plot


class Inspector(Base):

    def __init__(self):
        return super().__init__()

    def __post_inspect(self, data: Data, logger: Logger):

        records = data.x
        labels = data.y

        summary = {}
        summary['balance'] = aggregate.balance(labels)
        plot.labels_distribution(labels, logger, 'balance')

        plot.features_distribution(
            records,
            labels,
            logger,
            'distribution'
        )
