from abc import ABC, abstractproperty, abstractmethod

from lazy_property import LazyProperty


class DataSet(ABC):
    """
    Abstract container for data set.

    Parameters
    ----------
    ABC : class
        Module from Python's standard lib,
        used to implement abstract classes.

    """

    def __init__(self, path: str):
        """
        Base abstract class for data sets.

        Parameters
        ----------
        path : str
            Path to your data set. Be it exact file, folder, doesn't matter,
            loading will be handled by specific `DataSet` implementation.
        """

        super().__init__()
        self.path = path

    @abstractproperty
    def name(self):
        """
        Name of your data set.
        You can easily extract it from path or folder name.
        """

        pass

    @abstractproperty
    def records(self):
        """
        This prop should return array of records.
        """

        pass

    @abstractproperty
    def labels(self):
        """
        This prop should return array of labels.
        """

        pass

    @abstractmethod
    def dropna(self):
        pass

    @abstractmethod
    def drop_duplicates(self):
        pass

    @LazyProperty
    def pprint(self):
        return {
            'name': self.name,
        }
