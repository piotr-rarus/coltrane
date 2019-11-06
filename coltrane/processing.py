import itertools
from abc import ABC, abstractmethod
from multiprocessing import Pool
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, Iterator, List

from austen import Logger
from colorama import init
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from coltrane import Batch, util
from coltrane.file.io.base import Data

init()


class Processor(ABC):

    def __init__(self):
        return super().__init__()

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

    def process(self, batches: Iterator[Batch], output: Path):

        batch: Batch

        for batch in tqdm(batches, desc='Batches'):
            batch.pprint()

            logs_dir = Path(output, batch.data.name, batch.as_nice_hash)
            with Logger(logs_dir) as logger:

                logger.save_json(batch.as_dict, 'batch')

                stats = self._process_batch(batch, logger)

                logger.add_entry('summary', stats.aggregated_metrics)
                logger.add_entry('performance', stats.aggregated_performance)
                util.plot.metrics(stats.grouped_metrics, logger)

    def __aggregate_batch(
        self,
        batch_stats: Dict[str, util.pipeline.Stats],
        logger: Logger
    ):
        grouped = {}

        with logger.get_child('aggregate') as logger:
            for pipeline, stats in batch_stats.items():
                for metric, values in stats.grouped_metrics.items():

                    if metric not in grouped:
                        grouped[metric] = {}

                    grouped[metric][pipeline] = values

            for metric, pipelines in grouped.items():
                util.plot.metrics(pipelines, logger, plot_name=metric)

    def _process_batch(
        self,
        batch: Batch,
        logger: Logger
    ) -> util.pipeline.Stats:

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
            metrics = batch.metrics
            encoder = batch.encoder

            train_X = data.X[train_index]
            train_y = data.y[train_index]
            train_y = encoder.transform(train_y)

            test_X = data.X[test_index]
            test_y = data.y[test_index]
            test_y = encoder.transform(test_y)

            start = timer()
            pipeline.fit(train_X, train_y)
            end = timer()
            dt_fit = end - start

            start = timer()
            pred_y = pipeline.predict(test_X)
            end = timer()
            dt_predict = end - start
            dt_predict_record = dt_predict / len(pred_y)

            performance = utility.split.Performance(
                dt_fit,
                dt_predict,
                dt_predict_record
            )

            evaluation = utility.metric.evaluate(test_y, pred_y, metrics)

            logger.add_entry('metrics', evaluation)
            logger.add_entry('performance', performance.as_dict())
            logger.save_obj(pipeline, 'pipeline')

            self.__post_split(data, test_y, pred_y, logger)

            return utility.split.Stats(evaluation, performance)
