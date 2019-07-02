from pathlib import Path

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from coltrane import Batch, file
from coltrane.classification import Inspector, Processor

__LOGS = Path('logs')
__DATA_IRIS = Path('tests/data/iris.csv')

__RANDOM_STATE = 45625461


def get_metrics():
    return [
            accuracy_score,
            (
                precision_score,
                {
                    'average': 'macro'
                }
            ),
            (
                recall_score,
                {
                    'average': 'macro'
                }
            ),
            (
                f1_score,
                {
                    'average': 'macro'
                }
            )
        ]


def pipelines():
    yield Pipeline(
        steps=[
            ('naive-bayes', GaussianNB())
        ]
    )

    yield Pipeline(
        steps=[
            ('standard-scaler', StandardScaler()),
            ('svc', SVC(gamma='auto'))
        ]
    )


def batches():

    yield Batch(
        data_set=file.io.csv.single.DataSet(path=__DATA_IRIS),
        pipelines=pipelines,
        selection=RepeatedKFold(
            n_splits=5,
            n_repeats=2,
            random_state=__RANDOM_STATE
        ),
        metrics=get_metrics()
    )


def data_set():
    yield file.io.csv.single.DataSet(path=__DATA_IRIS)


def test_inspection():
    inspector = Inspector()
    inspector.inspect(data_set, output=__LOGS)


def test_classification():
    processor = Processor()
    processor.process(batches, output=__LOGS)
