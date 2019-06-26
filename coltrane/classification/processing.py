from austen import Logger
from sklearn.metrics import confusion_matrix as get_confusion_matrix

from ..processing import Processor as BaseProcessor
from .. import utility
from ..file.io.base import DataSet


class Processor(BaseProcessor):

    def __init__(self):
        return super().__init__()

    def __post_split(
        self,
        data_set: DataSet,
        test_y,
        pred_y,
        logger: Logger,
        *args,
        **kwargs
    ):

        confusion_matrix = get_confusion_matrix(test_y, pred_y)

        labels = list(dict.fromkeys(data_set.y).keys())

        utility.plot.confusion_matrix(confusion_matrix, labels, logger)

    def __post_batch(
        self,
        batch_stats: utility.batch.Stats,
        logger: Logger,
        *args,
        **kwargs
    ):
        pass
