from pathlib import Path

from lazy_property import LazyProperty, LazyWritableProperty

from . import base


class DataSet(base.DataSet):
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

    @LazyWritableProperty
    def y(self):
        return self.__extract_y()

    def __extract_y(self):
        return self.__data_set.iloc[:, -1].values

    def as_dict(self):
        base = super().as_dict()
        base['type'] = __name__,
        return base
