from austen import Logger

from coltrane.file.io.base import Data
from coltrane.inspection import Inspector as Base


class Inspector(Base):

    def __init__(self):
        return super().__init__()

    def __post_inspect(self, data: Data, logger: Logger):
        pass

    def plot_target_distribution(self, data: Data):
        """
        Plots target distribution into the notebook.

        Parameters
        ----------
        data : Data
            Your data set.
        """

        self.plot.distribution(data.y, 'Target Distribution')
