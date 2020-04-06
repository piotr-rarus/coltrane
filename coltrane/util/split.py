from dataclasses import dataclass


@dataclass(init=True)
class Performance():
    dt_fit: float
    # dt_predict: float
    # dt_predict_record: float

    def as_dict(self):
        return self.__dict__


@dataclass(init=True)
class Stats():
    metrics: dict
    performance: Performance
