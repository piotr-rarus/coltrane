from austen import Logger

from coltrane.file.io.base import Data
from coltrane.inspection import Inspector as Base
from coltrane.util import aggregate


class Inspector(Base):

    def __init__(self):
        super(Inspector, self).__init__()

    def __post_inspect(self, data: Data, logger: Logger):

        records = data.x
        labels = data.y

        summary = {}
        balance = aggregate.balance(labels)
        summary['balance'] = balance

        self.plot.class_balance(balance)

        self.plot.features_distribution(records, labels)

        return summary
