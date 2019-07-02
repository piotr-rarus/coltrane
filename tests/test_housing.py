from pathlib import Path

import sklearn.metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from coltrane import Batch, file
from coltrane.regression import Inspector, Processor
from xgboost import XGBRegressor

__LOGS = 'logs'
__DATA_HOUSING = Path('tests/data/housing.csv')

__RANDOM_STATE = 45625461


def pipelines():

    yield Pipeline(
        steps=[
            ('linear', LinearRegression())
        ]
    )

    yield Pipeline(
        steps=[
            ('ridge', Ridge())
        ]
    )

    yield Pipeline(
        steps=[
            ('kernel-ridge', KernelRidge())
        ]
    )

    yield Pipeline(
        steps=[
            ('svr', SVR(gamma='scale'))
        ]
    )

    yield Pipeline(
        steps=[
            ('xgboost', XGBRegressor())
        ]
    )


def batches():

    yield Batch(
        data_set=file.io.csv.single.DataSet(path=__DATA_HOUSING),
        pipelines=pipelines,
        selection=RepeatedKFold(
            n_splits=5,
            n_repeats=5,
            random_state=__RANDOM_STATE
        ),
        metrics=[
            sklearn.metrics.r2_score,
            sklearn.metrics.mean_squared_error
        ]
    )


def data_set():
    yield file.io.csv.single.DataSet(path=__DATA_HOUSING)


def test_inspection():
    inspector = Inspector()
    inspector.inspect(data_set, output=__LOGS)


def test_regression():
    processor = Processor()
    processor.process(batches, output=__LOGS)
