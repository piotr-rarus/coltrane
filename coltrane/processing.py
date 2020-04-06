import itertools
from abc import ABC, abstractmethod
from multiprocessing import Pool
from pathlib import Path
from timeit import default_timer as timer
from typing import List

from austen import Logger
from colorama import init
from tqdm import tqdm

from coltrane import Batch, util
from coltrane.util import Plot
from coltrane.file.io.base import Data

init()


class Processor(ABC):

    def __init__(self):
        super(Processor, self).__init__()
        self.plot = Plot()

    @abstractmethod
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

    @abstractmethod
    def __post_pipeline(
        self,
        stats: util.pipeline.Stats,
        logger: Logger,
        *args,
        **kwargs
    ):
        pass

    def process(self, batch: Batch, output: Path) -> util.pipeline.Stats:

        # batch.pprint()

        log_dir = Path(output, batch.data.name, batch.as_nice_hash)
        with Logger(log_dir) as logger:

            logger.save_json(batch.as_dict, 'batch')

            stats = self._process_batch(batch, logger)

            logger.add_entry('summary', stats.aggregated_scores)
            logger.add_entry('performance', stats.aggregated_performance)
            self.plot.metrics(stats.grouped_scores)

            return stats

    def _process_batch(
        self,
        batch: Batch,
        logger: Logger
    ) -> util.pipeline.Stats:

        if batch.encoder:
            batch.encoder.fit(batch.data.y)
            logger.save_obj(batch.encoder, 'encoder')

        splits = batch.selection.split(batch.data.x, batch.data.y)

        splits = [
            (batch, split_index, train_index, test_index, logger) for
            split_index, (train_index, test_index) in enumerate(splits)
        ]

        splits_iter = tqdm(splits, desc='Splits')

        starmap = itertools.starmap

        if batch.multiprocessing:
            pool = Pool()
            starmap = pool.starmap

        splits_stats = starmap(self._process_split, splits_iter)

        pipeline_stats = util.pipeline.Stats(list(splits_stats))
        self.__post_pipeline(pipeline_stats, logger)

        return pipeline_stats

    def _process_split(
        self,
        batch: Batch,
        split_index: int,
        train_index: List[int],
        test_index: List[int],
        logger: Logger
    ) -> util.split.Stats:

        with logger.get_child(str(split_index)) as logger:

            data = batch.data
            pipeline = batch.pipeline
            scorers = batch.scorers
            encoder = batch.encoder

            x_train = data.x[train_index]
            y_train = data.y[train_index]

            x_test = data.x[test_index]
            y_test = data.y[test_index]

            if encoder:
                y_train = encoder.transform(y_train)
                y_test = encoder.transform(y_test)

            start = timer()
            pipeline.fit(x_train, y_train)
            end = timer()
            dt_fit = end - start

            performance = util.split.Performance(
                dt_fit,
                # dt_predict,
                # dt_predict_record
            )

            scores = util.metric.evaluate(scorers, pipeline, x_test, y_test)

            logger.add_entry('scores', scores)
            logger.add_entry('performance', performance.as_dict())
            logger.save_obj(pipeline, 'pipeline')

            # self.__post_split(data, y_test, pred_y, logger)

            return util.split.Stats(scores, performance)
