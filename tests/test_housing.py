from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, max_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from coltrane import Batch, csv
from coltrane.regression import Inspector, Processor

__LOGS = 'logs'
__DATA_HOUSING = 'tests\\data\\housing.csv'

__RANDOM_STATE = 45625461


def batches():

    yield Batch(
        data=csv.single_label.DataSet(path=__DATA_HOUSING),
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
        metrics=[
            r2_score
        ]
    )

    yield Batch(
        data=csv.single_label.DataSet(path=__DATA_HOUSING),
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
        metrics=[
            r2_score
        ]
    )


def data_set():
    yield csv.single_label.DataSet(path=__DATA_HOUSING)


def test_inspection():
    inspector = Inspector()
    inspector.inspect(data_set, output=__LOGS)


def test_regression():
    processor = Processor()
    processor.process(batches, output=__LOGS)
