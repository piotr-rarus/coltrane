from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef
)
from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from coltrane import Batch, csv
from coltrane.classification import Inspector, Processor

__LOGS = 'logs'
__DATA_IRIS = 'tests\\data\\iris.csv'

__RANDOM_STATE = 45625461


def metrics():
    return [
            accuracy_score,
            (
                precision_score,
                {
                    'average': 'micro'
                }
            ),
            (
                recall_score,
                {
                    'average': 'micro'
                }
            ),
            (
                f1_score,
                {
                    'average': 'micro'
                }
            ),
            (
                matthews_corrcoef,
                {

                }
            )
        ]


def batches():

    yield Batch(
        data=csv.single.DataSet(path=__DATA_IRIS),
        pipeline=Pipeline(
            steps=[
                ('standard-scaler', StandardScaler()),
                ('naive-bayes', GaussianNB())
            ]
        ),
        selection=RepeatedKFold(
            n_splits=5,
            n_repeats=2,
            random_state=__RANDOM_STATE
        ),
        metrics=metrics()
    )

    yield Batch(
        data=csv.single.DataSet(path=__DATA_IRIS),
        pipeline=Pipeline(
            steps=[
                ('standard-scaler', StandardScaler()),
                ('svc', SVC(gamma='auto'))
            ]
        ),
        selection=RepeatedKFold(
            n_splits=5,
            n_repeats=2,
            random_state=__RANDOM_STATE
        ),
        metrics=metrics()
    )


def data_set():
    yield csv.single.DataSet(path=__DATA_IRIS)


def test_inspection():
    inspector = Inspector()
    inspector.inspect(data_set, output=__LOGS)


def test_classification():
    processor = Processor()
    processor.process(batches, output=__LOGS)
