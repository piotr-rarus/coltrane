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

    def __init__(self, path, encoder):
        """
        Initiates data set.

        Parameters
        ----------
        path : string
            Path to your data set `.csv` file.
        encoder : obj
            Reference to encoder that should be used to encode labels.

        """
        super().__init__(path, encoder)

    @LazyWritableProperty
    def labels(self):
        return self.__extract_labels()

    def __extract_labels(self):
        labels = self.__data_set.iloc[:, -1].values
        return list([label.split(' ') for label in labels])

    @LazyProperty
    def pprint(self):
        base = super().pprint
        base['type'] = __name__,
        return base
