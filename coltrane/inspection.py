import json
import os
from abc import ABC, abstractmethod
from typing import Generator

from austen import Logger
from colorama import init
from tqdm import tqdm

from .file.io.base import DataSet

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
            self.__pprint_data(data_set)

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

                self.__calc_stats(data_set, logger)

    def __pprint_data(self, data_set: DataSet):
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

    @abstractmethod
    def __calc_stats(self, data_set: DataSet, logger: Logger):
        pass
