from abc import ABC, abstractmethod
from timeit import default_timer as timer
from typing import Generator, List

from austen import Logger
from colorama import init
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from pathlib import Path

from . import utility
from .batch import Batch
from .file.io.base import DataSet


init()


class Processor(ABC):

    def __init__(self):
        return super().__init__()

    @abstractmethod
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

    @abstractmethod
    def __post_pipeline(
        self,
        stats: utility.pipeline.Stats,
        logger: Logger,
        *args,
        **kwargs
    ):
        pass

    # @abstractmethod
    # def __post_batch(
    #     self,
    #     batch_stats: utility.batch.Stats,
    #     logger: Logger,
    #     *args,
    #     **kwargs
    # ):
    #     pass

    def process(
        self,
        batches: Generator[Batch, None, None],
        output: Path
    ):

        for batch in tqdm(batches(), desc='Batches'):
            batch.pprint()

            batch_dir = Path(output, batch.data_set.name)
            with Logger(batch_dir) as batch_logger:
                self.__process_batch(batch, batch_dir, batch_logger)

    def __process_batch(
        self,
        batch: Batch,
        batch_dir: Path,
        logger: Logger
    ):
        for pipeline in tqdm(batch.pipelines(), desc='Pipelines'):

            pipeline_hash = str(hash(pipeline))
            pipeline_dir = batch_dir.joinpath(pipeline_hash)

            with Logger(pipeline_dir) as pipeline_logger:

                batch_as_dict = batch.as_dict()

                batch_as_dict['pipeline'] = self.__pipeline_as_dict(pipeline)
                pipeline_logger.save_json(batch_as_dict, 'batch')

                self.__process_pipeline(
                    batch.data_set,
                    batch.selection,
                    pipeline,
                    batch.metrics,
                    pipeline_logger
                )

    def __pipeline_as_dict(self, pipeline: Pipeline):
        as_dict = {}

        for step in pipeline:
            as_dict[step.__class__.__name__] = vars(step)

        return as_dict

    def __process_pipeline(
        self,
        data_set: DataSet,
        selection,
        pipeline: Pipeline,
        metrics,
        logger: Logger
    ):
        splits = selection.split(data_set.X, data_set.y)
        splits_iter = enumerate(tqdm(splits, desc='Splits'))

        stats = utility.pipeline.Stats()

        with logger.get_child('splits') as splits_logger:
            # TODO: generator doesn't have length attribute
            # TODO: so there's kinda lame progressbar :/
            # ? does all selectors implement `n_splits` prop?
            for split_index, (train_index, test_index) in splits_iter:
                with splits_logger.get_child(str(split_index)) as split_logger:

                    split_stats = self.__process_split(
                        data_set,
                        pipeline,
                        metrics,
                        train_index,
                        test_index,
                        split_logger
                    )

                    stats.splits.append(split_stats)

            logger.add_entry(
                'summary',
                stats.aggregated_metrics
            )

            logger.add_entry(
                'performance',
                stats.aggregated_performance
            )

            utility.plot.metrics(stats.grouped_metrics, logger)

            self.__post_pipeline(stats, logger)

        return stats

    def __process_split(
        self,
        data_set: DataSet,
        pipeline: Pipeline,
        metrics,
        train_index: List[int],
        test_index: List[int],
        logger: Logger
    ) -> utility.split.Stats:

        train_X = data_set.X[train_index]
        train_y = data_set.y[train_index]

        test_X = data_set.X[test_index]
        test_y = data_set.y[test_index]

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
        logger.add_entry('performance', performance.as_dict())
        logger.save_obj(pipeline, 'pipeline')

        self.__post_split(data_set, test_y, pred_y, logger)

        return utility.split.Stats(
            evaluation,
            performance
        )
