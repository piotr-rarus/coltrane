from austen import Logger

from .. import utility
from ..processing import Processor as BaseProcessor
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
        pass

    def __post_pipeline(
        self,
        stats: utility.pipeline.Stats,
        logger: Logger,
        *args,
        **kwargs
    ):
        pass
