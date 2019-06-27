import json
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path

from lazy_property import LazyProperty
from tqdm import tqdm


class DataSet(ABC):
    """
    Abstract container for data set.

    Parameters
    ----------
    ABC : class
        Module from Python's standard lib,
        used to implement abstract classes.

    """

    def __init__(self, path: Path):
        """
        Base abstract class for data sets.

        Parameters
        ----------
        path : Path
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
    def data(self):
        pass

    @abstractproperty
    def X(self):
        """
        This prop should return array of records.
        """

        pass

    @abstractproperty
    def y(self):
        """
        This prop should return array of ground truth values.
        """

        pass

    @abstractmethod
    def dropna(self):
        pass

    @abstractmethod
    def drop_duplicates(self):
        pass

    @LazyProperty
    def as_dict(self):
        return {
            'name': self.name,
        }

    def pprint(self):
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
        tqdm.write(json.dumps(self.as_dict, indent=4))
        tqdm.write('\n' * 3)
