from pathlib import Path

from pytest import fixture
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from coltrane import Batch
from coltrane.file.io.csv.single import Data
from coltrane.regression import Inspector, Processor

__LOG = Path('log')
__DATA_HOUSING = Path('test/data/housing.csv')

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
        scorers={
            'r2': make_scorer(r2_score),
            'mse': make_scorer(mean_squared_error)
        }
    )


def test_inspection(data: Data):
    inspector = Inspector()
    inspector.inspect(data, output=__LOG)


def test_regression(batch: Batch):
    processor = Processor()
    processor.process(batch, output=__LOG)
