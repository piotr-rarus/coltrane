from ..processing import Processor as BaseProcessor
from sklearn.metrics import confusion_matrix as get_confusion_matrix
from ..utility import batch, plot
from austen import Logger


class Processor(BaseProcessor):

    def __init__(self):
        return super().__init__()

    def __post_split(self, test_y, pred_y, logger: Logger, *args, **kwargs):

        confusion_matrix = get_confusion_matrix(
            test_y,
            pred_y,
            labels=kwargs.labels
        )

        plot.confusion_matrix(confusion_matrix, kwargs.labels, logger)

    def __post_batch(batch_stats: batch.Stats, *args, **kwargs):
        pass
