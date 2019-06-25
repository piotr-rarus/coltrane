import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Generator

import numpy as np
from austen import Logger
from colorama import init
from tqdm import tqdm

from .file.io.base import DataSet
from .utility import plot

init()


class Inspector(ABC):

    def __init__(self):
        return super().__init__()

    def inspect(self, data: Generator[DataSet, None, None], output: str):
        """
        Gain some insights from your data.
        This a priori knowledge will help you configure meaningful pipelines
        and help you prepare your data.

        Parameters
        ----------
        data : Generator[DataSet, None, None]
            Should yield instantiated `base.DataSet` objects.
        output : str
            Points folder, where logs with stats will be dumped.

        """

        for data_set in tqdm(data(), desc='Data'):
            data_set.pprint()

            output = os.path.join(
                output,
                data_set.name,
                'inspection'
            )

            with Logger(output) as logger:
                logger.add_entry('data', data_set.as_dict)
                self.__inspect(data_set, logger)

    def __inspect(self, data_set: DataSet, logger: Logger):

        X = data_set.X

        summary = OrderedDict()
        X_count, attributes_count = X.shape

        summary['records'] = OrderedDict()
        summary['records']['count'] = X_count

        missing_values = np.count_nonzero(data_set.isna())
        summary['records']['missing-values'] = missing_values

        summary['attributes'] = OrderedDict()
        summary['attributes']['count'] = attributes_count

        categorical_attributes = self.__get_categorical_attributes(data_set)
        summary['attributes']['categorical'] = categorical_attributes

        numerical_attributes = self.__get_numerical_attributes(data_set)
        summary['attributes']['numerical'] = numerical_attributes

        (
            pearson,
            kendall,
            spearman
        ) = self.__calc_correlation_maps(data_set, logger)

        description = data_set.describe().to_dict()
        logger.save_json(description, 'description')

        logger.add_entry('summary', summary)

        self.__post_inspect(data_set, logger)


    @abstractmethod
    def __post_inspect(self, data_set: DataSet, logger: Logger):
        pass

    def __get_numerical_attributes(self, data_set: DataSet):
        numerical = []

        for attribute, values in zip(data_set.attributes, data_set.X.T):
            any_categorical = self.__any_categorical(values)

            if not any_categorical:
                numerical.append(attribute)

        return numerical

    def __get_categorical_attributes(self, data_set: DataSet):
        categorical = []

        for attribute, values in zip(data_set.attributes, data_set.X.T):
            any_categorical = self.__any_categorical(values)

            if any_categorical:
                categorical.append(attribute)

        return categorical

    def __any_categorical(self, values):
        return any(type(value) is str for value in values)

    def __calc_correlation_maps(self, data_set: DataSet, logger: Logger):
        data = data_set.data

        pearson = data.corr(method='pearson')
        plot.heatmap(pearson, logger, 'pearson')

        kendall = data.corr(method='kendall')
        plot.heatmap(kendall, logger, 'kendall')

        spearman = data.corr(method='spearman')
        plot.heatmap(spearman, logger, 'spearman')

        return pearson, kendall, spearman
