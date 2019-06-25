from dataclasses import dataclass

from sklearn.pipeline import Pipeline


@dataclass(init=True)
class Performance():
    dt_fit: float
    dt_predict: float
    dt_predict_record: float

    def to_dict(self):
        return self.__dict__


@dataclass(init=True)
class Stats():
    pipeline: Pipeline
    metrics: dict
    performance: Performance
