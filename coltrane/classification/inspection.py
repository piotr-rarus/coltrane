from typing import Dict

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

        return summary

    def plot_features_distribution(self, data: Data):
        self.plot.features_distribution(data.x, data.y)

    def get_class_balance(self, data: Data):
        labels = data.y

        balance = aggregate.balance(labels)
        return balance

    def plot_class_balance(self, balance: Dict[str, int]):
        self.plot.class_balance(balance)
