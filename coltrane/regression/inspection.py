from austen import Logger

from coltrane.file.io.base import Data
from coltrane.inspection import Inspector as Base
from coltrane.util import plot


class Inspector(Base):

    def __init__(self):
        return super().__init__()

    def __post_inspect(self, data: Data, logger: Logger):
        y = data.y

        # TODO
        # plot.distribution(y, logger, 'target distribution')
