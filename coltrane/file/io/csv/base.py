from abc import abstractmethod, abstractproperty
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd
from lazy import lazy

from .. import base


class Data(base.Data):
    """
    Handles old school data sets.
    Data set should be .csv table.
    First row for the header.
    Ground truth value in the last column.
    First column for the record ID.
    """

    def __init__(self, path: Path):
        """
        Base abstract class for `csv` based data.

        Parameters
        ----------
        path : Path
            Path to your data `csv` file.
        """

        super().__init__(path)

    @lazy
    def __data_set(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    @lazy
    def xy(self) -> pd.DataFrame:
        return self.__data_set.iloc[:, 1:]

    @lazy
    def name(self) -> str:
        return self.path.stem

    @lazy
    def x(self):
        return self.__extract_X()

    def __extract_X(self):
        return self.__data_set.iloc[:, 1:-1].values

    @abstractproperty
    def y(self):
        pass

    @abstractmethod
    def __extract_y(self):
        pass

    @lazy
    def attributes(self):
        return self.__data_set.columns[1:-1].values

    @lazy
    def as_dict(self) -> Dict:
        as_dict = super().as_dict
        as_dict['path'] = self.path.name
        return as_dict

    def isna(self) -> np.ndarray:
        """
        Detect missing values.

        > Return a boolean same-sized object indicating if the values are NA.
        > NA values, such as None or numpy.NaN, gets mapped to True values.
        > Everything else gets mapped to False values.
        > Characters such as empty strings '' or numpy.inf are not considered
        > NA values (unless you set pandas.options.mode.use_inf_as_na = True).

        Returns
        -------
        nd.array[bool]
            Return a boolean same-sized object indicating if the values are NA.
        """

        return self.__data_set.isna()

    def __preprocess(func: Callable):

        def wrapper(self, *args, **kwargs):
            func(self)
            self.x = self.__extract_X()
            self.y = self.__extract_y()
            return

        return wrapper

    @__preprocess
    def dropna(self):
        """
        Drops records, that hold missing feature value.

        """

        self.__data_set = self.__data_set.dropna()

    @__preprocess
    def drop_duplicates(self):
        """
        Drops every duplicate occurrence.
        Considering same records, that share also class, first occurrence
        will be kept.
        Considering same records, but when class differs, none will be kept.

        """

        # ? This will keep first of duplicated records, that share class.
        columns = self.__data_set.columns[1:]
        self.__data_set.drop_duplicates(subset=columns, inplace=True)

        # ? This will drop duplicate records, that doesn't share class.
        # ? It's considered to be contradictory knowledge.
        self.__data_set.drop_duplicates(
            subset=self.attributes,
            keep=False,
            inplace=True
        )

    def describe(self):
        features = self.__data_set[self.attributes]
        return features.describe()
