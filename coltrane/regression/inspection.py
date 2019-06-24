from austen import Logger

from ..file.io.base import DataSet
from ..inspection import Inspector as Base
from ..utility import plot


class Inspector(Base):

    def __init__(self):
        return super().__init__()

    def __post_inspect(self, data_set: DataSet, logger: Logger):
        # data = data_set.data
        # X = data_set.X
        y = data_set.y

        plot.distribution(y, logger, 'target distribution')
