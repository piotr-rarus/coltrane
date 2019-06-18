from glob import glob
from typing import List, Generator

from . import base


class DataSets():

    def __init__(self, folder, data_set, encoder):
        self.__files = self.__extract_files(folder)

        if not self.__files:
            raise DirectoryEmpty()

        self.__data_set = data_set
        self.__encoder = encoder

    def __extract_files(self, folder) -> List[str]:
        return glob(folder + '*.csv')

    def generate(self) -> Generator[base.DataSet, None, None]:
        for file in self.__files:
            yield self.__data_set(file, self.__encoder)


class DirectoryEmpty(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.description = "Folder directory contains no `.csv` files"
