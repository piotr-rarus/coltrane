from pathlib import Path
from typing import List

from pytest import fixture
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from coltrane import Batch
from coltrane.classification import Inspector, Processor
from coltrane.file.io.csv.single import Data


__LOGS = Path('logs')
__DATA_IRIS = Path('tests/data/iris.csv')
__RANDOM_STATE = 45625461


@fixture(scope='function')
def data() -> Data:
    return Data(path=__DATA_IRIS)


@fixture(scope='function')
def pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ('standard-scaler', StandardScaler()),
            ('naive-bayes', GaussianNB())
        ]
    )


@fixture(scope='session')
def selection():
    return RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=1,
        random_state=__RANDOM_STATE
    )


@fixture(scope='session')
def metrics():
    return [
            accuracy_score,
            (
                precision_score,
                {
                    'average': 'weighted'
                }
            ),
            (
                recall_score,
                {
                    'average': 'weighted'
                }
            ),
            (
                f1_score,
                {
                    'average': 'weighted'
                }
            )
        ]


@fixture(scope='function')
def batch(data: Data, pipeline: Pipeline, selection, metrics) -> Batch:

    return Batch(
        data,
        pipeline,
        selection,
        metrics,
        encoder=LabelEncoder(),
        multiprocessing=True
    )


def test_inspection(data: Data):
    inspector = Inspector()
    inspector.inspect([data], output=__LOGS)


def test_classification(batch: Batch):
    processor = Processor()
    processor.process([batch], output=__LOGS)
