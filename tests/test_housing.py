from pathlib import Path

import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from coltrane import Batch, file
from coltrane.regression import Inspector, Processor

__LOGS = 'logs'
__DATA_HOUSING = Path('tests/data/housing.csv')

__RANDOM_STATE = 45625461


def get_metrics():
    return [
        (
            sklearn.metrics.explained_variance_score,
            {
                'multioutput': 'variance_weighted'
            }
        ),
        sklearn.metrics.r2_score,
        sklearn.metrics.mean_squared_error
    ]


def batches():

    yield Batch(
        data=file.io.csv.single.DataSet(path=__DATA_HOUSING),
        pipeline=Pipeline(
            steps=[
                ('standard-scaler', StandardScaler()),
                ('naive-bayes', LinearRegression())
            ]
        ),
        selection=RepeatedKFold(
            n_splits=5,
            n_repeats=2,
            random_state=__RANDOM_STATE
        ),
        metrics=get_metrics()
    )

    yield Batch(
        data=file.io.csv.single.DataSet(path=__DATA_HOUSING),
        pipeline=Pipeline(
            steps=[
                ('standard-scaler', StandardScaler()),
                ('svc', SVR())
            ]
        ),
        selection=RepeatedKFold(
            n_splits=5,
            n_repeats=2,
            random_state=__RANDOM_STATE
        ),
        metrics=get_metrics()
    )


def data_set():
    yield file.io.csv.single.DataSet(path=__DATA_HOUSING)


def test_inspection():
    inspector = Inspector()
    inspector.inspect(data_set, output=__LOGS)


def test_regression():
    processor = Processor()
    processor.process(batches, output=__LOGS)
