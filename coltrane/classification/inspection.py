import json
import os
from typing import Generator

import numpy as np
from austen import Logger
from colorama import init
from tqdm import tqdm

from ..file.io.base import DataSet
from ..utility import aggregate, plot

init()


def inspect(data: Generator[DataSet, None, None], output: str):
    """
    Gain some insights from your data.
    This a priori knowledge will help you configure meaningful pipelines.
    Stats include:
        - type of problem
        - attribute vector length
        - class balance
        - classes vs records
        - type of attributes (categorical/numerical)
        - missing values and sparsity

    Parameters
    ----------
    data : Generator[DataSet, None, None]
        Should yield instantiated `base.DataSet` objects.
    output : str
        Points folder, where logs with stats will be dumped.

    """

    for data_set in tqdm(data(), desc='Data'):
        __pprint_data(data_set)

        output = os.path.join(
            output,
            data_set.name,
            'inspection'
        )

        with Logger(output) as logger:
            logger.add_entry('data', data_set.pprint)

            logger.add_entry(
                'attributes',
                {
                    'count': len(data_set.attributes),
                    'values': data_set.attributes
                }
            )

            __dump_stats(data_set, logger, '0-raw-data')

            data_set.dropna()
            __dump_stats(data_set, logger, '1-drop-missing')

            data_set.drop_duplicates()
            __dump_stats(data_set, logger, '2-drop-duplicates')

            plot.features_distribution(
                data_set.records,
                data_set.labels,
                logger,
                'features-distribution'
            )


def __dump_stats(data_set: DataSet, logger: Logger, step: str):
    description = data_set.describe()
    logger.save_csv(description, step + '-description')

    records = data_set.records
    labels = data_set.labels

    plot.labels_distribution(labels, logger, step + '-balance')

    stats = {}
    records_count, attributes_count = records.shape

    stats['balance'] = aggregate.balance(labels)

    stats['records'] = {}
    stats['records']['count'] = records_count
    stats['records']['attributes-ratio'] = records_count / attributes_count

    missing_values = np.count_nonzero(data_set.isna())
    stats['records']['missing-values'] = missing_values
    stats['records']['missing-values-ratio'] = missing_values / records_count

    numerical = []
    categorical = []

    for attribute, feature in zip(data_set.attributes, records.T):
        is_categorical = any(type(value) is str for value in feature)

        if is_categorical:
            categorical.append(attribute)
        else:
            numerical.append(attribute)

    stats['attributes-categorical'] = categorical
    stats['attributes-numerical'] = numerical

    logger.add_entry(step, stats)


def __pprint_data(data_set: DataSet):
    """
    Pretty prints your data set.

    Parameters
    ----------
    data_set : DataSet
        Instantiated data set.

    """

    tqdm.write('\n' * 3)
    tqdm.write('=' * 100)
    tqdm.write('\n' * 3)

    tqdm.write('Data set:')
    tqdm.write(json.dumps(data_set.pprint, indent=4))
    tqdm.write('\n' * 3)
