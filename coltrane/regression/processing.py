from ..processing import Processor as BaseProcessor
from .. import utility

from austen import Logger


class Processor(BaseProcessor):

    def __init__(self):
        return super().__init__()

    def __post_split(self, test_y, pred_y, logger: Logger, *args, **kwargs):
        pass

    def __post_batch(batch_stats: utility.batch.Stats, *args, **kwargs):
        pass
