from pathlib import Path
from lazy import lazy

from . import base


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
        Initiates data set.

        Parameters
        ----------
        path : Path
            Path to your data `csv` file.
        """

        super().__init__(path)

    @lazy
    def y(self):
        return self.__extract_y()

    def __extract_y(self):
        return self.__data_set.iloc[:, -1].values

    @lazy
    def as_dict(self):
        base = super().as_dict
        base['type'] = __name__,
        return base
