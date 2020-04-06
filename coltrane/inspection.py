from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from austen import Logger
from colorama import init

from coltrane.file.io.base import Data
from coltrane.util import Plot

init()


class Inspector(ABC):

    def __init__(self):
        super(Inspector, self).__init__()

        self.plot = Plot()

    def inspect(self, data: Data, output: Path):
        """
        Gain some insights from your data.
        This a priori knowledge will help you configure meaningful pipelines
        and help you prepare your data.

        Parameters
        ----------
        data : Data
            Your data set.
        output : str
            Points folder, where logs with stats will be dumped.

        """

        # data.pprint()
        log_dir = Path(output, data.name, 'inspection')

        with Logger(log_dir) as logger:
            logger.add_entry('data', data.as_dict)

            x = data.x

            summary = {}
            x_count, attributes_count = x.shape

            summary['records'] = {}
            summary['records']['count'] = x_count

            missing_values = np.count_nonzero(data.isna())
            summary['records']['missing-values'] = missing_values

            summary['attributes'] = {}
            summary['attributes']['count'] = attributes_count

            categorical_attributes = self.__get_categorical_attributes(data)
            summary['attributes']['categorical'] = categorical_attributes

            numerical_attributes = self.__get_numerical_attributes(data)
            summary['attributes']['numerical'] = numerical_attributes

            summary['post'] = self.__post_inspect(data, logger)

            (
                pearson,
                kendall,
                spearman
            ) = self.__calc_correlation_maps(data, logger)

            description = data.describe().to_dict()
            logger.save_json(description, 'description')

            logger.add_entry('summary', summary)


    @abstractmethod
    def __post_inspect(self, data: Data, logger: Logger):
        pass

    def __get_numerical_attributes(self, data: Data):
        numerical = []

        for attribute, values in zip(data.attributes, data.x.T):
            any_categorical = self.__any_categorical(values)

            if not any_categorical:
                numerical.append(attribute)

        return numerical

    def __get_categorical_attributes(self, data: Data):
        categorical = []

        for attribute, values in zip(data.attributes, data.x.T):
            any_categorical = self.__any_categorical(values)

            if any_categorical:
                categorical.append(attribute)

        return categorical

    def __any_categorical(self, values):
        return any(type(value) is str for value in values)

    def __calc_correlation_maps(self, data: Data, logger: Logger):
        data = data.xy

        pearson = data.corr(method='pearson')
        self.plot.heatmap(pearson, 'Pearson')

        kendall = data.corr(method='kendall')
        self.plot.heatmap(kendall, 'Kendall')

        spearman = data.corr(method='spearman')
        self.plot.heatmap(spearman, 'Spearman')

        return pearson, kendall, spearman
