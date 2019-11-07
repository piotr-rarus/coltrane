from austen import Logger

from coltrane import util
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

    def __post_pipeline(
        self,
        stats: util.pipeline.Stats,
        logger: Logger,
        *args,
        **kwargs
    ):
        pass
