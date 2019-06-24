import numpy as np
from austen import Logger

from ..file.io.base import DataSet
from ..inspection import Inspector as Base
from ..utility import aggregate, plot


class Inspector(Base):

    def __init__(self):
        return super().__init__()

    def __calc_stats(self, data_set: DataSet, logger: Logger):
        description = data_set.describe().to_dict()
        logger.save_json(description, 'description')

        records = data_set.records
        labels = data_set.labels

        # TODO: histogram

        stats = {}
        records_count, attributes_count = records.shape

        stats['records'] = {}
        stats['records']['count'] = records_count

        missing_values = np.count_nonzero(data_set.isna())
        stats['records']['missing-values'] = missing_values

        numerical = []
        categorical = []

        for attribute, feature in zip(data_set.attributes, records.T):
            is_categorical = any(type(value) is str for value in feature)

            if is_categorical:
                categorical.append(attribute)
            else:
                numerical.append(attribute)

        stats['attributes-categorical'] = categorical
        stats['attributes-numerical'] = numerical

        logger.add_entry('stats', stats)

        # TODO: plot histogram
