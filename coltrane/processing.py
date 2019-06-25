import json
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from timeit import default_timer as timer
from typing import Generator, List

from austen import Logger
from colorama import init
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from . import utility
from .batch import Batch
from .file.io.base import DataSet

init()


class Processor(ABC):

    def __init__(self):
        return super().__init__()

    @abstractmethod
    def __post_split(self, test_y, pred_y, logger: Logger, *args, **kwargs):
        pass

    @abstractmethod
    def __post_batch(batch_stats: utility.batch.Stats, *args, **kwargs):
        pass

    def process(
        self,
        batches: Generator[Batch, None, None],
        output: str
    ):
        for batch in tqdm(batches(), desc='Pipelines'):
            batch.pprint()

            logs_dir = self.__get_output(batch.data, output)

            with Logger(logs_dir) as logger:

                logger.save_json(batch.pprint, 'batch')
                batch_stats = self.__process_batch(batch, logger)

    def __get_output(self, data_set: DataSet, output: str):
        now = datetime.now()

        return os.path.join(
            output,
            data_set.name,
            str(now.timestamp())
        )

    def __process_batch(self, batch: Batch, logger: Logger):
        data = batch.data
        selection = batch.selection
        pipeline = batch.pipeline
        metrics = batch.metrics

        splits = selection.split(data.X, data.y)
        splits_iter = enumerate(tqdm(splits, desc='Splits'))

        batch_stats = utility.batch.Stats()

        with logger.get_child('splits') as splits_logger:
            # TODO: generator doesn't have length attribute
            # TODO: so there's kinda lame progressbar :/
            # ? does all selectors implement `n_splits` prop?
            for split_index, (train_index, test_index) in splits_iter:
                with splits_logger.get_child(str(split_index)) as split_logger:

                    split_stats = self.__process_split(
                        data,
                        pipeline,
                        metrics,
                        train_index,
                        test_index,
                        split_logger
                    )

                    batch_stats.splits.append(split_stats)

            logger.add_entry(
                'summary',
                batch_stats.get_aggregated_metrics()
            )

            logger.add_entry(
                'performance',
                batch_stats.get_aggregated_performance()
            )

            utility.plot.metrics(batch_stats.grouped_metrics, logger)

            self.__post_batch(batch_stats)

        return batch_stats

    def __process_split(
        self,
        data: DataSet,
        pipeline: Pipeline,
        metrics,
        train_index: List[int],
        test_index: List[int],
        logger: Logger
    ) -> utility.split.Stats:

        train_X = data.X[train_index]
        train_y = data.y[train_index]

        test_X = data.X[test_index]
        test_y = data.y[test_index]

        # TODO: this looks ugly, maybe some wrapper?
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
        logger.add_entry('performance', performance.__dict__)
        logger.save_obj(pipeline, 'pipeline')

        self.__post_split(test_y, pred_y, set(data.y), logger)

        return utility.split.Stats(
            deepcopy(pipeline),
            metrics,
            performance
        )
