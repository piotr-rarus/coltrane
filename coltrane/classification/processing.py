from ..processing import Processor as BaseProcessor
from sklearn.metrics import confusion_matrix as get_confusion_matrix
from ..utility import plot
from austen import Logger


class Processor(BaseProcessor):

    def __init__(self):
        return super().__init__()

    def __post_split(self, test_y, pred_y, labels, logger: Logger):
        labels = list(labels)

        confusion_matrix = get_confusion_matrix(
            test_y,
            pred_y,
            labels=labels
        )

        plot.confusion_matrix(confusion_matrix, labels, logger)

    def __post_batch(self):
        pass
