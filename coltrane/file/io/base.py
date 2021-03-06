import json
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Dict

from lazy import lazy
from tqdm.auto import tqdm


class Data(ABC):
    """
    Abstract container for data set.
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
    def xy(self):
        pass

    @abstractproperty
    def x(self):
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

    @lazy
    def as_dict(self) -> Dict:
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

        tqdm.write('Data set:')
        tqdm.write(json.dumps(self.as_dict, indent=4))
        tqdm.write('\n' * 3)
