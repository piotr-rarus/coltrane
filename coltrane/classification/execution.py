import json
import os
from datetime import datetime
from typing import Generator

from austen import Logger
from colorama import init
from tqdm import tqdm

from .template import process
from ..file.io.base import DataSet
from ..batch import Batch

init()


def execute(
    batches: Generator[Batch, None, None],
    output: str
):
    # ! docs

    for batch in tqdm(batches(), desc='Pipelines'):
        __pprint_research(batch)

        logs_dir = __get_output(batch.data, output)

        with Logger(logs_dir) as logger:

            logger.add_entry('batch', batch.pprint)
            process(batch, logger)


def __get_output(data_set: DataSet, output: str):
    now = datetime.now()

    return os.path.join(
        output,
        data_set.name,
        str(now.timestamp())
    )


def __pprint_research(batch: Batch):
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
