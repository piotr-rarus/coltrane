from lazy_property import LazyProperty, LazyWritableProperty

from . import base


class DataSet(base.DataSet):
    """
    Handles old school data sets.
    Data set should be .csv table.
    First row for the header.
    Class label in the last column
    First column for the record ID.
    """

    def __init__(self, path):
        """
        Initiates data set.

        Parameters
        ----------
        path : string
            Path to your data `csv` file.
        """

        super().__init__(path)

    @LazyWritableProperty
    def labels(self):
        return self.__extract_labels()

    def __extract_labels(self):
        return self.__data_set.iloc[:, -1].values

    @LazyProperty
    def pprint(self):
        base = super().pprint
        base['type'] = __name__,
        return base
