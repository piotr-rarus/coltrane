from austen import Logger
# from sklearn.metrics import confusion_matrix as get_confusion_matrix

from coltrane.processing import Processor as BaseProcessor
from coltrane.file.io.base import Data


class Processor(BaseProcessor):

    def __init__(self):
        return super().__init__()

    def __post_split(
        self,
        data_set: Data,
        test_y,
        pred_y,
        logger: Logger,
        *args,
        **kwargs
    ):
        pass
        # labels = list(set(data_set.y))
        # self.__plot_confusion_matrix(test_y, pred_y, labels, logger)

    # def __plot_confusion_matrix(
    #     self,
    #     test_y,
    #     pred_y,
    #     labels,
    #     logger: Logger
    # ):
    #     confusion_matrix = get_confusion_matrix(test_y, pred_y)

    #     self.plot.heatmap(
    #         confusion_matrix,
    #         'Confusion matrix',
    #         labels,
    #         labels
    #     )
