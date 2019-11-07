from pathlib import Path

import sklearn.metrics
from pytest import fixture
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from coltrane import Batch
from coltrane.file.io.csv.single import Data
from coltrane.regression import Inspector, Processor


__LOGS = 'logs'
__DATA_HOUSING = Path('tests/data/housing.csv')

__RANDOM_STATE = 45625461


@fixture(scope='function')
def data() -> Data:
    return Data(path=__DATA_HOUSING)


@fixture(scope='function')
def pipeline():
    return Pipeline(
        steps=[
            ('robust-scaler', RobustScaler()),
            ('linear', LinearRegression())
        ]
    )


@fixture(scope='session')
def selection():
    return RepeatedKFold(
        n_splits=5,
        n_repeats=1,
        random_state=__RANDOM_STATE
    )


@fixture(scope='function')
def batch(data: Data, pipeline: Pipeline, selection) -> Batch:

    return Batch(
        data,
        pipeline,
        selection,
        metrics=[
            sklearn.metrics.r2_score,
            sklearn.metrics.mean_squared_error
        ]
    )


def test_inspection(data: Data):
    inspector = Inspector()
    inspector.inspect([data], output=__LOGS)


def test_regression(batch: Batch):
    processor = Processor()
    processor.process([batch], output=__LOGS)
