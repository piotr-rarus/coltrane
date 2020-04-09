import itertools
from abc import ABC, abstractmethod
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

from austen import Logger
from sklearn.metrics._scorer import _BaseScorer
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from coltrane import Batch
from coltrane.file.io.base import Data
from coltrane.util import Plot
from coltrane.util.stats import BatchStats, SplitStats


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

    def process(self, batch: Batch, output: Path) -> BatchStats:

        # batch.pprint()

        log_dir = Path(output, batch.data.name, batch.as_nice_hash)
        with Logger(log_dir) as logger:

            logger.save_json(batch.as_dict, 'batch')

            stats = self._process_batch(batch, logger)

            self.plot.scores(stats.grouped_scores)

            return stats

    def _process_batch(
        self,
        batch: Batch,
        logger: Logger
    ) -> BatchStats:

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

        return BatchStats(list(splits_stats))

    def _process_split(
        self,
        batch: Batch,
        split_index: int,
        train_index: List[int],
        test_index: List[int],
        logger: Logger
    ) -> SplitStats:

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

            pipeline.fit(x_train, y_train)

            scores = self.__evaluate_metrics(scorers, pipeline, x_test, y_test)

            logger.add_entry('scores', scores)
            logger.save_obj(pipeline, 'pipeline')

            # self.__post_split(data, y_test, pred_y, logger)

            return SplitStats(scores, deepcopy(pipeline))

    def __evaluate_metrics(
        self,
        scorers: Dict[str, _BaseScorer],
        pipeline: Pipeline,
        x_test,
        y_test
    ):
        stats = {}

        for name, scorer in scorers.items():

            score = scorer(pipeline, x_test, y_test)
            stats[name] = score

        return stats
