import json
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from timeit import default_timer as timer
from typing import Generator, List

import numpy as np
from austen import Logger
from colorama import init
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from .batch import Batch
from .file.io.base import DataSet
from .utility import aggregate, plot

init()


@dataclass(init=True)
class SplitPerformance():
    dt_fit: float
    dt_predict: float
    dt_predict_record: float

    def to_dict(self):
        return self.__dict__


class Performance():

    def __init__(self):
        self.dt_fit = []
        self.dt_predict = []
        self.dt_predict_record = []

    def append(self, split: SplitPerformance):
        self.dt_fit.append(split.dt_fit)
        self.dt_predict.append(split.dt_predict)
        self.dt_predict_record.append(split.dt_predict_record)

    def aggregate(self):
        stats = OrderedDict()

        for stat in self:
            stats[stat.__name__] = self.__aggregate_stat(stat)

        return stats

    def __aggregate_stat(values: List[float]):
        return {
            'mean': np.mean(values),
            'min': np.min(values),
            'max': np.max(values),
            'std': np.std(values)
        }


class Processor(ABC):

    def __init__(self):
        return super().__init__()

    @abstractmethod
    def __post_split(self, test_y, pred_y, labels, logger: Logger):
        pass

    @abstractmethod
    def __post_batch():
        pass

    def process(
        self,
        batches: Generator[Batch, None, None],
        output: str
    ):
        for batch in tqdm(batches(), desc='Pipelines'):
            self.__pprint_research(batch)

            logs_dir = self.__get_output(batch.data, output)

            with Logger(logs_dir) as logger:

                logger.save_json(batch.pprint, 'batch')
                self.__process_batch(batch, logger)

    def __pprint_research(self, batch: Batch):
        """
        Pretty prints your data set and pipeline onto console using tqdm.

        Parameters
        ----------
        data_set : DataSet
            Instantiated data set.
        pipeline : Pipeline
            Configured pipeline template.

        """

        tqdm.write('\n' * 3)
        tqdm.write('=' * 100)
        tqdm.write('\n' * 3)

        tqdm.write('Data set:')
        tqdm.write(json.dumps(batch.pprint, indent=4))
        tqdm.write('\n' * 3)

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

        stats = []

        performance = Performance()

        with logger.get_child('splits') as splits_logger:
            # TODO: generator doesn't have length attribute
            # TODO: so there's kinda lame progressbar :/
            # ? does all selectors implement `n_splits` prop?
            for split_index, (train_index, test_index) in splits_iter:
                with splits_logger.get_child(str(split_index)) as split_logger:

                    evaluation, split_performance = self.__process_split(
                        data,
                        pipeline,
                        metrics,
                        train_index,
                        test_index,
                        split_logger
                    )

                    stats.append(evaluation)
                    performance.append(split_performance)

            grouped_stats = aggregate.group_metrics(stats)

            aggregated_metrics = aggregate.stats(grouped_stats)
            aggregated_performance = aggregate.performance(performance)

            logger.add_entry('summary', aggregated_metrics)
            logger.add_entry('performance', aggregated_performance)

            plot.metrics(grouped_stats, logger)

        self.__post_batch()

    def __process_split(
        self,
        data: DataSet,
        pipeline: Pipeline,
        metrics,
        train_index: List[int],
        test_index: List[int],
        logger: Logger
    ):
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

        performance = SplitPerformance(dt_fit, dt_predict, dt_predict_record)

        evaluation = self.__evaluate_metrics(test_y, pred_y, metrics)

        logger.add_entry('metrics', evaluation)
        logger.add_entry('performance', performance.__dict__)
        logger.save_obj(pipeline, 'pipeline')

        self.__post_split(test_y, pred_y, set(data.y), logger)

        return evaluation, performance

    def __evaluate_metrics(self, test, pred, metrics):
        stats = OrderedDict()

        for metric in metrics:
            name, value = self.__evaluate_metric(test, pred, metric)
            stats[name] = value

        return stats

    def __evaluate_metric(self, test, pred, metric):
        op, kwargs = None, None

        if type(metric) is tuple and len(metric) == 2:
            op, kwargs = metric

        elif callable(metric):
            op = metric
            kwargs = {}

        value = op(test, pred, **kwargs)

        name = op.__name__

        return name, value

    def __update_performance(self, performance, telemetry):
        for key, value in telemetry.items():
            performance[key].append(value)
